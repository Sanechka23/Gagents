from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def load_recipes(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    recipes = [line.strip() for line in data if line.strip() and "Little Alchemy" not in line]
    return recipes

def create_knowledge_base(recipes):
    documents = [Document(page_content=recipe) for recipe in recipes]
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def iterative_find_recipe(query, vectorstore, available_items, max_iterations=3):
    """Находит рецепт, выполняя пошаговый поиск недостающих элементов."""

    chat_model = ChatGroq(
        temperature=0.1,
        model_name="llama3-70b-8192",
        groq_api_key=""
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = (
        "У тебя есть словарь уравнений, к которому ты можешь обращаться\n"
        "Представь, что все слова в нем это не слова, а математические переменные, сотстоящие из русских букв."
        "Внутри словаря каждая строчка это уровнение с тремя матемаическими перменными.\n"
        "У тебя также искомая математичсекая перменная '{query}'. "
        "а также исходные математические перменные '{available_items}', записанные через запятую.\n\n"
        "Твоя задача - выразить искомую математичсекая перменная через исходные математические перменные.\n\n"
        "Чтобы обратиться к словарю, отправь команду для поиска информации, которая начинается исключительно с 'SEARCH:'."
        "Тебе НЕЛЬЗЯ выдумывать ответ на команду поиска\n"
        "Результат команды поиска:{context}"
        "Тебе НЕЛЬЗЯ повторно искать одну и ту же переменные в словаре.\n"
        "Ответом должно быть уравнение\n"
        "Размышляй на русском языке.\\n"
        "Пример твоей работы:\n"
        "Искомая математичсекая перменная = Арфа. Исходные математические перменные  = человек, птица, музыка.\n"
        "Ты отправляюешь команду поиска: SEARCH: Арфа =, на что получаешь ответ из словаря 'Арфа = музыка + ангел'.\n"
        "После этого ты сверяешь исходные математические перменные с отвтеом из словаря, и замечаешь, что перменная 'музыка' совпадает."
        "Значит нужно найти перменную 'ангел'.\n"
        "ты отправляешь команду для поиска в виде ответа: SEARCH: Ангел =, на что получаешь ответ из словаря 'Ангел = человек + птица'.\n"
        "После этого ты сверяешь исходные математические перменные с отвтеом из словаря, и замечаешь, что перменные 'человек' и 'птица' совпадают."
        "Значит надо проверить, что все переменные выраженны через исходные математические перменные: 'Арфа = ангел + музыка = человек + птица + музыка'.\n"
        "Дейтсвтительно, в последней части уравнения остались только исходные математические перменны, значит, можно написать ответ: ' Арфа = человек + птица + музыка'."


    )

    context = ""
    final_answer = None

    for iteration in range(1, max_iterations + 1):
        formatted_prompt = prompt_template.format(
            available_items=", ".join(available_items),
            context=context,
            query=query
        )
        print(f"\n--- Итерация {iteration} ---")
        print("Промпт для LLM:\n", formatted_prompt)

        prompt_obj = PromptTemplate(template=formatted_prompt, input_variables=["dummy"])
        chain = LLMChain(llm=chat_model, prompt=prompt_obj)

        response = chain.run({"dummy": ""})
        print("Ответ LLM:\n", response)

        if response.strip().upper().find("SEARCH:") != -1:
            print(1)
            match = re.match(r"SEARCH:\s*(.*)", response.strip(), re.IGNORECASE)
            if match:
                search_query = match.group(1).strip()
                print("Команда поиска обнаружена. Поиск по запросу:", search_query)
                docs = retriever.get_relevant_documents(search_query)
                additional_context = "\n".join(doc.page_content for doc in docs)
                print("Найденные рецепты:\n", additional_context)
                context += "\n" + additional_context
                continue
            else:
                final_answer = response.strip()
                break
        elif response.strip().upper().startswith("ANSWER:"):
            final_answer = response.strip()[len("ANSWER:"):].strip()
            break
        else:
            final_answer = response.strip()
            break

    if final_answer is None:
        final_answer = "Не удалось получить окончательный ответ."
    return final_answer

def main():
    file_path = "recipes.txt"
    recipes = load_recipes(file_path)
    vectorstore = create_knowledge_base(recipes)

    available_items = ["собака", "жизнь", "земля", "лес"]
    query = "Оборотень"

    answer = iterative_find_recipe(query, vectorstore, available_items)
    print("\nОкончательный ответ:\n", answer)

if __name__ == "__main__":
    main()


# Второй промт:
#     "Ты эксперт по игре Little Alchemy. Твоя задача – составить подробный пошаговый рецепт создания элемента.\n"
#     "Размышляй и отвечай ТОЛЬКО на русском языке.\n "
#     "никогда не выдумывай рецепты самостоятельно, используя рецепты ТОЛЬКО из базы данных.\n"
#     "Если тебе не хватает информации, чтобы составить полную цепочку, выдай команду для поиска, начни ответ со строки 'SEARCH:' и затем задай уточняющий запрос.\n"
#     "Если же у тебя достаточно информации, выдай окончательный ответ, начиная со строки 'ANSWER:' и затем свой ответ."
#     "Также в окончательном ответе должны быть только цепочки рецептов после ANSWER:.\n"
#     "Пример итогового ответа: 'ANSWER:\n 1)Арфа = ангел + музыка\n 2)Ангел = человек + птица'.\n"
#     "Доступные элементы: {available_items}.\n"
#     "Известные рецепты:\n{context}\n"
#     "Вопрос: Как создать {query}?\n"
#     "Ответ:"