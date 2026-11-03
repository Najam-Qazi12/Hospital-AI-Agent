# AI Hospital Receptionist

A chat-based AI receptionist that manages hospital appointments, built with LangGraph, Groq and SQLite.

## Features
- Book, find, reschedule and cancel appointments
- Blocks double-booking (same doctor, same date and time)
- Doctor info, fees and hospital timings
- Conversation memory
- Safety rules: no medical advice, emergencies redirected

## Tech
Python, LangGraph, LangChain, Groq (gpt-oss-20b), SQLite

## Run it
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your Groq API key
3. `python hospital_agent.py`

## Demo


https://github.com/user-attachments/assets/bc010325-b439-4f40-8129-840f1b1f9614






## Known limits / next steps
- One appointment per patient name (a real system would use patient ID)
- Time slots are free text; fixed slots would be more reliable
- Next: Streamlit web UI
