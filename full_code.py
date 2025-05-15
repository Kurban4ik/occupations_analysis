import json
import pandas as pd
import time
import requests
from openai import OpenAI
from tqdm import tqdm
import os
from time import sleep

# подключаем данные для обработки
data = pd.read_excel('occupation.xlsx')
# объединим столбцы-дубликаты
data['occupation'] = data['job_title_or_prof_130'].combine_first(data['job_title_or_prof'])
data['plwrk'] = data['work_place_130'].combine_first(data['work_place'])

def clean(data, col_to_clean='occupation', res_col='clear'):
    """
    добавляет к data столбец res_col, образованный после очистки столбца column_to_clean
    data: pd.DataFrame
    column_to_clean: str
    res_col: str
    :return: None
    """
    def delete_trash(occ):
        out_occ = ''
        for i in occ:
            if i == ' ':
                if len(out_occ) != 0 and out_occ[-1] != ' ':
                    out_occ += ' '
            else:
                if i.isalpha():
                    out_occ += i
                else:
                    if len(out_occ) != 0 and out_occ[-1] != ' ':
                        out_occ += ' '
        return out_occ.strip().lower()
    unique_values = data[col_to_clean].dropna().str.strip().str.lower().unique()
    cleaning_map = {val: delete_trash(val) for val in unique_values}
    data[res_col] = data[col_to_clean].dropna().str.lower().str.strip().map(cleaning_map)

def yandexspeller(data, col_to_fix='clear', res_col='fixed_yandex', batch_size=50):
    """
    Исправляет ошибки в словах столбца col_to_fix и сохраняет результат в res_col
    
    важное замечание: обработка ведется с помощью API яндекса у которого ограничение
    на 10000 запросов в день с одного адреса.
    
    обработка же ведется формируя батчи для исправления,
    где зависимость такая: больше батч --> быстрее обработается и меньше запросов, но
                           больше батч --> больше неправильно обработанных.
    data: pd.DataFrame
    col_to_fix: str
    res_col: str
    batch_size: int
    :return: None
    """
    
    non_na = data[col_to_fix].dropna()
    unique_values = non_na.unique()
    def correct_phrases(phrases, lang='ru'):
        url = "https://speller.yandex.net/services/spellservice.json/checkTexts"
        params = {
            'text': phrases,  # Список фраз
            'lang': lang,
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return phrases  # Если запрос не удался, возвращаем исходные фразы
        results = response.json()
        corrected_phrases = []
        for i, phrase in enumerate(phrases):
            errors = results[i]  # Ошибки для текущей фразы
            if not errors:
                corrected_phrases.append(phrase)  # Если ошибок нет, оставляем фразу как есть
                continue
            corrected_phrase = phrase
            for error in reversed(errors):  # Идем с конца, чтобы не сбить индексы
                start = error['pos']  # Позиция начала ошибки
                length = error['len']  # Длина ошибочного слова
                end = start + length
                corrected_word = error['s'][0] if error['s'] else error['word']
                corrected_phrase = corrected_phrase[:start] + corrected_word + corrected_phrase[end:]
            corrected_phrases.append(corrected_phrase)
        return corrected_phrases
    ar = list(unique_values)
    cleaning_map = {}
      # лучше выбирать малые пачки для обработки, так как апи начинает пропускать ошибки при увеличении батча
    for i in range(batch_size, len(ar) + batch_size - 1, batch_size):
        r = correct_phrases(ar[i - batch_size:i])
        print(r[0])
        for j in range(len(r)):
            cleaning_map[ar[i - batch_size + j]] = r[j].replace('ё', 'е')
        time.sleep(0.3)
    data[res_col] = data[col_to_fix].map(cleaning_map)

clean(data, col_to_clean='occupation')
yandexspeller(data, col_to_fix='clear')
f = open('profs.txt', 'w', encoding='utf-8')
for i in list(data['fixed_yandex'].dropna().unique()):
    f.write(i+'\n')
def deepseekAPI():
    # API-key устарел
    client = OpenAI(api_key="api key", base_url="https://api.deepseek.com")
    
    with open('profs.txt', 'r', encoding='utf-8') as f:
        professions = [line.strip() for line in f if line.strip()]
    
    # Группировка по 20 профессий (меньше = стабильнее, но медленнее)
    group_size = 20
    profession_groups = [professions[i:i + group_size] for i in range(0, len(professions), group_size)]
    # Функция обработки группы
    def process_group(group):
        prompt = f"""
        Анализ профессий: {', '.join(group)}
        Верни JSON, где ключи - названия профессий, а значения - словари с:
        - "qualification_level" (1-10)
        - "hierarchy_level" (1-10)
        - "is_industrial" (0/1)
        - "is_healthcare" (0/1)
        - "is_management" (0/1)
        - "is_security" (0/1)
    
        Если параметр неопределим - ставь минимальное значение.
        Если это не профессия - все параметры -1.
        Только JSON без комментариев!
        """
    
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Ты аналитик профессий. Отвечай строго в JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return {}
    
    # Обработка с сохранением прогресса
    all_results = {}
    output_file = 'data_progress.json'
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    
    for group in tqdm(
            [g for g in profession_groups if ', '.join(g) not in all_results],
            desc="Обработка"
    ):
        group_key = ', '.join(group)
        result = process_group(group)
        print(result)
        if result:
            all_results[group_key] = result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            sleep(1)  # Задержка для избежания лимитов
    
    # Сбор финальных данных
    final_data = {}
    for group, data in all_results.items():
        for profession in group.split(', '):
            final_data[profession] = data.get(profession, {
                "qualification_level": -1,
                "hierarchy_level": -1,
                "is_industrial": -1,
                "is_healthcare": -1,
                "is_management": -1,
                "is_security": -1
            })
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! Обработано {len(final_data)} профессий.")
deepseekAPI()
def turn_into_features(data, types, col_name='correct_yandex'):
    """
    Эта функция насаживает для каждой профессии столбца col_name её фичи из types
    data: pd.DataFrame
    types: dict
    col_name: str
    :return: None
    """
    cols = ['qualification_level', 'hierarchy_level', 'is_industrial', 'is_healthcare', 'is_management', 'is_security']
    def get_features(row):
        if types.__contains__(row):
            v = list(types[row].values())
            return v
        else:
            print(row)
        return [None] * len(cols)
    non_na = data[col_name]
    unique_values = non_na.unique()
    features_map = {val: get_features(val) for val in unique_values}
    data[cols] = pd.DataFrame(data[col_name].map(features_map).tolist(), index=data.index)
# подключаем словарь "профессия: (свойства)"
with open('data.json', 'r', encoding='utf-8') as f:
  types = json.load(f)
turn_into_features(data, types, 'fixed_yandex')
def check_impute(col):  # Функция просто считает  количество невалидно отработанных записей
    void = 0
    non_void = 0
    for i in list(data[col].dropna()):
        if types.__contains__(i):
            if list(types[i].values())[0] != -1:
                non_void += 1
            else:
                void += 1
        else:
            void += 1
    print('incorrectly served or nan amount:', void)
    print('percentage of incorrectly served or nan amount', void / len(data[col]))
check_impute('fixed_yandex')
