import telebot
import re
from groq import Groq
from collections import defaultdict
import pandas as pd


client = Groq(api_key="")
bot = telebot.TeleBot("")


with open("Меню.txt", "r", encoding="utf-8") as f:
    menu_content = f.read()
menu_content = menu_content.replace("{", "{{").replace("}", "}}")

table_seats = pd.read_csv("table_seats.csv")
reservations = pd.read_csv("reservations.csv")


reservation_params = defaultdict(lambda: {"date": None, "time": None, "guest_count": None})


def retrieve_available_tables(date: str, time_input: str) -> list:
    fuzzy_times = {
        "утро": ("08:00", "12:00"),
        "днем": ("12:00", "18:00"),
        "вечером": ("18:00", "23:00")
    }
    if time_input in fuzzy_times:
        requested_start, requested_end = fuzzy_times[time_input]
    elif "-" in time_input:
        requested_start, requested_end = [s.strip() for s in time_input.split("-")]
    else:
        requested_start, requested_end = time_input, time_input

    matched_column = None
    for col in reservations.columns:
        if date.lower() in col.lower():
            matched_column = col
            break
    if not matched_column:
        return []

    available_tables = []
    for _, row in reservations.iterrows():
        slot = str(row[matched_column]).strip()
        if slot.lower() == "свободен":
            available_tables.append(row["Номер стола"])
        else:
            if requested_start and requested_end:
                def time_to_minutes(t: str) -> int:
                    h, m = map(int, t.split(":"))
                    return h * 60 + m
                req_start = time_to_minutes(requested_start)
                req_end = time_to_minutes(requested_end)
                try:
                    booked_start, booked_end = [s.strip() for s in slot.split("-")]
                except Exception:
                    continue
                booked_start_m = time_to_minutes(booked_start)
                booked_end_m = time_to_minutes(booked_end)
                if req_start < booked_end_m and req_end > booked_start_m:
                    continue
                else:
                    available_tables.append(row["Номер стола"])
            else:
                available_tables.append(row["Номер стола"])
    return available_tables


def check_table_capacity(guest_count: int, available_tables: list) -> bool:
    total_capacity = 0
    for table in available_tables:
        row = table_seats[table_seats["Номер стола"] == table]
        if not row.empty:
            total_capacity += int(row["Количество мест"].iloc[0])
    return total_capacity >= guest_count

SYSTEM_PROMPT = f"""
Ты ИИ-агент ресторана "Силаушу". Твоя задача — помогать с бронированием столиков и отвечать на вопросы.

**Инструкции:**
1. Всегда сначала извлекай параметры бронирования (дата, время, количество гостей).
2. Если параметры определены, немедленно вызови функции:
   - `retrieve_available_tables(дата, время)` для получения списка доступных столов.
   - `check_table_capacity(количество гостей, список столов)` для проверки вместимости.
3. Формируй ответ в формате:
   - CALL: retrieve_available_tables("дата", "время")
   - CALL: check_table_capacity(гости, [список столов])
4. Если вызываешь функции, используй точный формат. Не пытайся придумать результат.
5. Используй эмодзи для краткости и наглядности.

**Доступные инструменты:**
- `retrieve_available_tables(date, time)` → возвращает список номеров столов
- `check_table_capacity(guest_count, available_tables)` → возвращает bool (достаточно ли мест)

**Меню ресторана:** {menu_content}
"""


chat_histories = defaultdict(list)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    with open("photo_2025-02-02_06-59-29.jpg", "rb") as photo:
        bot.send_photo(message.chat.id, photo)
    welcome_msg = (
        "🍣 Привет! Я Чичи – твой ИИ-агент в ресторане Силаушу.\n"
        "Я помогу с бронированием столика, расскажу о меню и проверю доступность столов.\n"
    )
    bot.send_message(message.chat.id, welcome_msg)

@bot.message_handler(content_types=['text'])
def handle_message(message):
    user_id = message.from_user.id
    user_input = message.text.strip()


    chat_histories[user_id].append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_histories[user_id])

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.3
        )
        if not response or not response.choices:
            raise RuntimeError("Неверный ответ от ИИ-модели")
        final_reply = response.choices[0].message.content.strip()


        call_match = re.match(r"CALL: (\w+)\((.*)\)", final_reply)
        if call_match:
            function_name = call_match.group(1)
            args = eval(call_match.group(2))

            if function_name == "retrieve_available_tables":
                result = retrieve_available_tables(*args)
                bot.send_message(message.chat.id, f"Доступные столы: {result}")
            elif function_name == "check_table_capacity":
                result = check_table_capacity(*args)
                bot.send_message(message.chat.id, f"Достаточно мест: {'да' if result else 'нет'}")
        else:
            bot.send_message(message.chat.id, final_reply)

        chat_histories[user_id].append({"role": "assistant", "content": final_reply})
    except Exception as e:
        print(f"Error: {str(e)}")
        bot.send_message(message.chat.id, "⚠️ Произошла ошибка. Пожалуйста, попробуйте задать вопрос иначе.")

if __name__ == "__main__":
    bot.infinity_polling()
