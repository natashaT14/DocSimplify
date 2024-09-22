pip install python-telegram-bot
pip install pdfplumber
pip install transformers


import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import pdfplumber
from transformers import BartForConditionalGeneration, BartTokenizer
import os


model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text: 
                full_text += text
    return full_text

def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

async def start(update: Update, context):
    await update.message.reply_text("Hi! I'm a PDF summarizer bot. Send me a PDF file, and I'll summarize it for you.")

async def handle_document(update: Update, context):
    file = update.message.document
    if file.mime_type == 'application/pdf':
        file_path = f"{file.file_id}.pdf"

        
        new_file = await file.get_file()
        await new_file.download_to_drive(file_path)

        text = extract_text_from_pdf(file_path)

        
        if len(text) > 2000:  # Limit to 2000 characters
            await update.message.reply_text("The document is too long to summarize. Please provide a shorter PDF.")
        else:
            await update.message.reply_text("Generating summary... Please wait ⏳")
            summary = summarize_text(text)
            await update.message.reply_text(f"Here’s the summary:\n{summary}")

        os.remove(file_path)
    else:
        await update.message.reply_text("Please send a PDF file.")

def main():
    TOKEN = "7042726869:AAGSULatZVDi0nQjYMUvrEODOTSFPomf5ck" 
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))

    
    application.run_polling()


if __name__ == "__main__":
    main()
