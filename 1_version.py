import telebot
from groq import Groq
from collections import defaultdict
import pandas as pd

# Инициализация клиентов
client = Groq(api_key="")
bot = telebot.TeleBot("")

# Загрузка данных
with open("Меню.txt", "r", encoding="utf-8") as f:
    menu_content = f.read()

table_seats = pd.read_csv("table_seats.csv")
reservations = pd.read_csv("reservations.csv")
closed_items = pd.read_csv("closed_items.csv")

def get_tables_info() -> str:
    """Форматирует информацию о столах для промпта"""
    return "\n".join([f"Стол {row['Номер стола']} ({row['Количество мест']} мест)"
                     for _, row in table_seats.iterrows()])

def get_raw_reservations() -> str:
    """Возвращает сырые данные бронирования в текстовом формате"""
    reservations_text = []
    for _, row in reservations.iterrows():
        table_info = [f"Стол {row['Номер стола']}:"]
        for col in reservations.columns[1:]:
            table_info.append(f"{col}: {row[col]}")
        reservations_text.append("\n".join(table_info))
    return "\n\n".join(reservations_text)

def get_closed_items() -> str:
    """Форматирует закрытые позиции меню"""
    closed = closed_items['Закрытые позиции'].tolist()
    return "Закрыто: " + ", ".join(closed) if closed else "Все блюда доступны"

SYSTEM_PROMPT = f"""
Ты администратор ресторана "Силаушу". Отвечай ТОЛЬКО на вопросы, связанные с рестораном.

Используй эти данные:

=== МЕНЮ ===
{menu_content}

=== СТОЛЫ ===
{get_tables_info()}

=== БРОНИРОВАНИЯ ===
{get_raw_reservations()}

=== ЗАКРЫТЫЕ ПОЗИЦИИ ===
{get_closed_items()}

Правила ответа:
1. Для бронирования спроси: дату, время и число гостей
2. В таблице есть временные промежутки и надписи Свободен. Если пользователь хочет совершить бронирование на время, поподающее хотя бы от части во временной промежуток имеющийся в таблице, то ты должен ему отказать в бронирование
3. Учитывай, что "Свободен" = весь день доступен
4. Если время не указано - покажи все доступные слоты на день
5. Форматируй ответы лаконичным текстом
6. Нельзя выводить все меню в одном сообщение
7. Если пользователь пытается говорить о темах несзяванных с рестораном, стараяйся вежливо вернуть диалог в нужную тематику\n'
8. не забывай иногда добовлять эмодзи
"""

chat_histories = defaultdict(list)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    with open("photo_2025-02-02_06-59-29.jpg", 'rb') as photo:
        bot.send_photo(message.chat.id, photo)

    welcome_msg = (
        "🍣 Привет! Я Чичи - ваш помощник в кафе Силаушу!\n"
        "Могу помочь с:\n"
        "• Бронированием столика\n"
        "• Информацией о меню\n"
        "• Проверкой закрытых позиций\n\n"
    )
    bot.send_message(message.chat.id, welcome_msg)

@bot.message_handler(content_types=['text'])
def handle_message(message):
    user_id = message.from_user.id
    text = message.text.strip()

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": "user" if msg["is_user"] else "assistant", "content": msg["content"]}
              for msg in chat_histories[user_id][-4:]],
            {"role": "user", "content": text}
        ]

        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.3
        )

        if not response or not response.choices:
            raise RuntimeError("Неверный ответ от ИИ-модели")

        ai_response = response.choices[0].message.content

        sent_message = bot.send_message(message.chat.id, ai_response)

        chat_histories[user_id].extend([
            {"is_user": True, "content": text},
            {"is_user": False, "content": ai_response}
        ])

        chat_histories[user_id] = chat_histories[user_id][-14:]

    except Exception as e:
        print(f"Error: {str(e)}")
        if not message.from_user.is_bot: 
            bot.send_message(
                message.chat.id,
                "⚠️ Произошла ошибка. Пожалуйста, попробуйте задать вопрос иначе."
            )

if __name__ == "__main__":
    bot.infinity_polling()