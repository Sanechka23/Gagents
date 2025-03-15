import json
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def load_recipes(file_path):
    """
    Загружает рецепты из файла, исключая строки, содержащие 'Little Alchemy'.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    recipes = [line.strip() for line in data if line.strip() and "Little Alchemy" not in line]
    return recipes

def retrieve_recipe_by_element(query, recipes):
    """
    Ищет рецепт для элемента.
    Возвращает те строки, где до знака '=' указан именно искомый элемент (без учета регистра).
    Например, для query = "Волк" возвращает строки, где слева от '=' ровно слово "волк".
    """
    result = []
    for recipe in recipes:
        if "=" in recipe:
            left_side = recipe.split("=")[0].strip().lower()
            if left_side == query.lower():
                result.append(recipe)
    return result

def save_json(file_path, data):
    """
    Сохраняет данные в JSON-файл.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def extract_first_json(text):
    """
    Извлекает первую сбалансированную JSON-структуру из текста.
    Возвращает подстроку от первой '{' до соответствующей ей '}'.
    Если не найдено, возвращает пустую строку.
    """
    start = text.find('{')
    if start == -1:
        return ""
    count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[start:i+1]
    return text[start:]

def iterative_find_recipe(query, recipes, available_items, max_iterations=200):
    """
    Пошагово ищет рецепт получения элемента.

    Алгоритм:
      1. Найди рецепт для элемента "{query}" (рассматривай только рецепты, где слева от '=' указан именно "{query}").
      2. Если рецепт для "{query}" не найден в контексте, выведи JSON:
             {"action": "search", "search_query": "{query}"}
      3. Если найден рецепт вида:
             "{query} = Ингредиент1 + Ингредиент2 [+ ...]"
         то для каждого ингредиента действуй так:
            - Если ингредиент содержится в available_items, он считается полученным.
            - Если ингредиент отсутствует в available_items, независимо от того, найден для него рецепт или нет, выведи JSON:
                 {"action": "search", "search_query": "<название отсутствующего ингредиента>"}
      4. Только если для элемента "{query}" найден рецепт, и все его ингредиенты содержатся в available_items, составь итоговую цепочку шагов от available_items до "{query}" и выведи JSON:
             {"action": "answer", "chain": ["шаг 1", "шаг 2", ..., "{query}"]}
      5. Обдумывай всю имеющуюся информацию (available_items, контекст, query). Если есть сомнения – выбирай действие "search".

    Перед генерацией JSON подробно объясни свои рассуждения, заключив их в теги <thinking> и </thinking>.
    Возвращай только валидный JSON без дополнительных комментариев или пояснений.
    """
    chat_model = ChatGroq(
        temperature=0.1,
        model_name="llama3-70b-8192",
        groq_api_key="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"
    )

    prompt_template = (
        "Ты эксперт по игре Little Alchemy. Твоя задача – составить пошаговый рецепт для получения элемента, используя рецепты из базы данных.\n\n"
        "Доступные элементы: {available_items}\n"
        "Запрос: получить элемент \"{query}\"\n"
        "Контекст (ранее найденные рецепты): {context}\n\n"
        "Действуй строго пошагово по следующему алгоритму:\n\n"
        "1. Найди рецепт для элемента \"{query}\". Рассматривай только рецепты, где до знака '=' указан именно \"{query}\".\n\n"
        "2. Если рецепт для \"{query}\" не найден в контексте, выведи JSON:\n"
        "   {{\"action\": \"search\", \"search_query\": \"{query}\"}}\n\n"
        "3. Если найден рецепт вида:\n"
        "   \"{query} = Ингредиент1 + Ингредиент2 [+ ...]\"\n"
        "   для каждого ингредиента поступай следующим образом:\n"
        "      - Если ингредиент содержится в доступных элементах ({available_items}), он считается полученным.\n"
        "      - Если ингредиент отсутствует в доступных элементах и конексте, выведи JSON:\n"
        "             {{\"action\": \"search\", \"search_query\": \"<название отсутствующего ингредиента>\"}}\n\n"
        "4. Только если для элемента \"{query}\" найден рецепт и все его ингредиенты содержатся в available_items, составь итоговую цепочку шагов от available элементов до \"{query}\" и выведи JSON:\n"
        "   {{\"action\": \"answer\", \"chain\": [\"шаг 1\", \"шаг 2\", ..., \"{query}\"]}}\n\n"
        "5. Обдумывай всю имеющуюся информацию (available_items, контекст, query). Если есть сомнения – выбирай действие \"search\".\n\n"
        "Перед генерацией JSON подробно объясни свои рассуждения, заключив их в теги <thinking> и </thinking>. Размышляй только на русском языке.\n"
        "Возвращай только валидный JSON без дополнительных комментариев или пояснений."
    )

    # Изначально контекст – пустой.
    context = "Нет найденных рецептов"
    final_answer = None

    for iteration in range(1, max_iterations + 1):
        request_data = {
            "available_items": available_items,
            "context": context,
            "query": query
        }
        save_json("llm_request.json", request_data)
        print(f"\n--- Итерация {iteration} ---")
        print("Запрос для LLM (llm_request.json):", request_data)

        prompt_obj = PromptTemplate(
            template=prompt_template,
            input_variables=["available_items", "context", "query"]
        )
        chain_obj = LLMChain(llm=chat_model, prompt=prompt_obj)
        response = chain_obj.run({
            "available_items": ", ".join(available_items),
            "context": context,
            "query": query
        })
        print("Сырой ответ LLM:\n", response)

        # Извлекаем первую JSON-структуру из ответа
        json_response = extract_first_json(response)
        # Также выводим рассуждения, если они есть
        if "<thinking>" in response and "</thinking>" in response:
            thinking = response.split("<thinking>")[1].split("</thinking>")[0].strip()
            print("\nРассуждения LLM:", thinking)
        else:
            print("\nРассуждения LLM: отсутствуют")

        try:
            response_json = json.loads(json_response)
        except Exception as e:
            print("Ошибка при разборе JSON-ответа от LLM:", e)
            final_answer = "Ошибка: Ответ от LLM не является валидным JSON."
            break

        action = response_json.get("action", "").lower()

        if action == "search":
            search_query = response_json.get("search_query", "").strip()
            search_word = search_query.split()[0] if search_query else ""
            if not search_word:
                final_answer = "Ошибка: Не получено корректное слово для поиска."
                break
            print("Команда поиска получена. Используем слово для поиска:", search_query)
            matching_recipes = retrieve_recipe_by_element(search_query, recipes)
            if matching_recipes:
                print("Найденные рецепты (где слово слева от '='):")
                for rec in matching_recipes:
                    print("  -", rec)
                new_info = "\n".join([rec for rec in matching_recipes if rec not in context])
                if context == "Нет найденных рецептов" or context.startswith("Нет совпадений"):
                    context = new_info
                else:
                    context += "\n" + new_info
            else:
                print(f"Нет рецептов, где слева от '=' встречается слово: {search_word}")
                new_info = f"Нет совпадений для запроса: {search_word}"
                if context == "Нет найденных рецептов":
                    context = new_info
                else:
                    context += "\n" + new_info
            continue

        elif action == "answer":
            chain_steps = response_json.get("chain", [])
            if not isinstance(chain_steps, list):
                final_answer = "Ошибка: Поле 'chain' должно быть списком шагов."
                break

            final_answer = chain_steps
            save_json("final_answer.json", {"chain": final_answer})
            break

        else:
            final_answer = "Ошибка: Неизвестное действие, полученное от LLM."
            break

    if final_answer is None:
        final_answer = "Не удалось получить окончательный ответ."
    return final_answer

def main():
    file_path = "recipes.txt"  # Пример содержимого:
    # Пуля = порох + металл
    # Металл = огонь + камень
    # Порох = огонь + пыль
    recipes = load_recipes(file_path)

    available_items1 = ["огонь", "воздух", "вода", "земля"]
    available_items2 = ["огонь", "камень", "день", "кровь", "олень", "лодка", "жизнь", "земля", "собака", "музыка", "птица", "жизнь", "лес"]
    query = "человек"

    answer = iterative_find_recipe(query, recipes, available_items1)
    print("\nОкончательный ответ:")
    if isinstance(answer, list):
        for step in answer:
            print("  -", step)
    else:
        print(answer)

if __name__ == "__main__":
    main()
