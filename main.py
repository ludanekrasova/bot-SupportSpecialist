import telebot
from telebot import types
from datetime import datetime, timedelta
import os

from model import predict

dtn = datetime.now() + timedelta(hours=5) # Дата
bot = telebot.TeleBot('123') # Токен бота

# в режиме разработки выводится класс и схожесть, в обычном простой диалог
dev = True

admin = [123] # Тг id куда будут приходить логи

#=========================================================================================
# Бот
#=========================================================================================

def log(message, answer_class=None, answer=None, similarity=None):
    # записывает логи чата
    botlogfile = open('Bot.log', 'a', encoding="utf8")
    stats = os.stat('Bot.log')
    if stats.st_size >= 1e+8:  # При достижении 100 мб файл будет скинут в тг и удалён с сервера
        bot.send_document(admin, document=open("Bot.log", "rb"), caption=dtn.strftime('%Y-%m-%d %H:%M:%S'))
        os.remove('Bot.log')
    else:
        print("------", file=botlogfile)
        print(dtn.strftime('%Y-%m-%d %H:%M:%S'), file=botlogfile)
        print("Сообщение от {0} (id = {1}) \n {2}".format(message.from_user.first_name, str(message.from_user.id), message.text), file=botlogfile)
        if answer_class and answer and similarity:
            print("Ответ: {}\nКласс ответа: {}\nСхожесть: {}".format(answer, answer_class, similarity),
                  file=botlogfile)
        botlogfile.close()

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, этот бот поможет вам ответить на вопросы о GeekBrains.", reply_markup=types.ReplyKeyboardRemove())
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Просто напиши в сообщении свой вопрос.")
    else:
        # строка, которую вводит пользователь
        answer_class, category, answer, similarity = predict(message.text)

        markup = types.InlineKeyboardMarkup()

        if (similarity > 0.2) and (similarity < 0.6):
            btn1= types.InlineKeyboardButton(text='Да', callback_data='correct')
            btn2= types.InlineKeyboardButton(text='Нет', callback_data='incorrect')
            markup.add(btn1, btn2)

            bot.send_message(message.from_user.id, f"Это вопрос из категории: {category}?", parse_mode="Markdown", reply_markup=markup)

        else:
            if dev == True:
                bot.send_message(message.from_user.id, f"{answer} \n\n Класс ответа: {answer_class}   {category} \n\n Схожесть: {similarity}", parse_mode="Markdown", reply_markup=markup)
            else:
                bot.send_message(message.from_user.id, f"{answer} ", parse_mode="Markdown", reply_markup=markup)

            log(message, answer_class, answer, similarity)

    # Обработчик нажатия на кнопки
    @bot.callback_query_handler(func=lambda call: True)
    def callback_handler(call):
        # строка, которую вводит пользователь
        answer_class, category, answer, similarity = predict(call.message.text)

        markup = types.InlineKeyboardMarkup()
        callback = call.data
        if callback == "correct":
            if dev == True:
                bot.send_message(call.message.chat.id, f"{answer} \n\n Класс ответа: {answer_class}   {category} \n\n Схожесть: {similarity}", parse_mode="Markdown", reply_markup=markup)
            else:
                bot.send_message(call.message.chat.id, f"{answer} ", parse_mode="Markdown", reply_markup=markup)
        elif callback == "incorrect":
            bot.send_message(call.message.chat.id, f"Пожалуйста, сформулируйте вопрос более развернуто.", parse_mode="Markdown", reply_markup=markup)


bot.polling(none_stop=True, interval=0)