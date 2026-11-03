import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Current date, passed to the model through the system prompt
current_date = datetime.now().strftime("%A, %B %d, %Y")

# ===============================================================
# 1. DATABASE
# ===============================================================
db = sqlite3.connect("hospital.db", check_same_thread=False)

db.execute("""
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT UNIQUE NOT NULL,
    doctor_name TEXT,
    appointment_time TEXT
)
""")
db.commit()


def clean_name(name: str) -> str:
    """Store patient names in one format so every tool finds the same record."""
    return name.lower().strip()


def clean_doctor(name: str) -> str:
    """Store doctor names in one format so conflict checks always match."""
    return name.strip().title()


# ===============================================================
# 2. TOOLS
# ===============================================================
@tool
def book_appointment(patient_name: str, doctor_name: str, appointment_time: str):
    """Book a new appointment for a patient."""
    patient_name = clean_name(patient_name)
    doctor_name = clean_doctor(doctor_name)

    # Check if the doctor is already booked at the requested time
    conflict = db.execute(
        """
        SELECT patient_name FROM appointments
        WHERE doctor_name = ? AND appointment_time = ?
        """,
        (doctor_name, appointment_time),
    ).fetchone()

    if conflict:
        return f"Booking failed. {doctor_name} is already booked at {appointment_time}."

    try:
        db.execute(
            """
            INSERT INTO appointments (patient_name, doctor_name, appointment_time)
            VALUES (?, ?, ?)
            """,
            (patient_name, doctor_name, appointment_time),
        )
        db.commit()
        return (
            f"Appointment for {patient_name.title()} was booked successfully "
            f"with {doctor_name} at {appointment_time}."
        )
    except sqlite3.IntegrityError:
        return f"An appointment for {patient_name.title()} already exists."


@tool
def find_appointment(patient_name: str):
    """Find a patient's appointment by their name."""
    appointment = db.execute(
        """
        SELECT id, patient_name, doctor_name, appointment_time
        FROM appointments
        WHERE patient_name = ?
        """,
        (clean_name(patient_name),),
    ).fetchone()

    if appointment:
        return (
            f"ID: {appointment[0]}, "
            f"Patient: {appointment[1].title()}, "
            f"Doctor: {appointment[2]}, "
            f"Time: {appointment[3]}"
        )
    return f"No appointment found for {patient_name}."


@tool
def show_appointments():
    """Show all booked appointments in the hospital database."""
    appointments = db.execute(
        """
        SELECT id, patient_name, doctor_name, appointment_time
        FROM appointments
        ORDER BY id
        """
    ).fetchall()

    if not appointments:
        return "There are no appointments in the database right now."

    result = ""
    for appt in appointments:
        result += (
            f"ID: {appt[0]} | "
            f"Patient: {appt[1].title()} | "
            f"Doctor: {appt[2]} | "
            f"Time: {appt[3]}\n"
        )
    return result


@tool
def update_appointment(patient_name: str, new_doctor: str, new_time: str):
    """Change a patient's doctor and appointment time."""
    patient_name = clean_name(patient_name)
    new_doctor = clean_doctor(new_doctor)

    # Check if the new slot is already taken by someone else
    conflict = db.execute(
        """
        SELECT patient_name FROM appointments
        WHERE doctor_name = ? AND appointment_time = ? AND patient_name != ?
        """,
        (new_doctor, new_time, patient_name),
    ).fetchone()

    if conflict:
        return f"Update failed. {new_doctor} is already booked at {new_time}."

    result = db.execute(
        """
        UPDATE appointments
        SET doctor_name = ?, appointment_time = ?
        WHERE patient_name = ?
        """,
        (new_doctor, new_time, patient_name),
    )
    db.commit()

    if result.rowcount == 0:
        return f"Appointment for {patient_name.title()} was not found."

    return f"{patient_name.title()}'s appointment was updated to {new_doctor} at {new_time}."


@tool
def cancel_appointment(patient_name: str):
    """Cancel and delete a patient's appointment from the database."""
    patient_name = clean_name(patient_name)

    result = db.execute(
        "DELETE FROM appointments WHERE patient_name = ?",
        (patient_name,),
    )
    db.commit()

    if result.rowcount == 0:
        return f"Appointment for {patient_name.title()} was not found."

    return f"Appointment for {patient_name.title()} was canceled successfully."


@tool
def get_doctor_info(search_query: str = "all"):
    """Get doctor details. Pass a doctor's name, a specialty, or 'all' to see everyone."""
    doctors = {
        "dr. smith": {"specialty": "Cardiologist", "price": "Rs. 5,000"},
        "dr. adam": {"specialty": "Dermatologist", "price": "Rs. 4,000"},
        "dr. sarah": {"specialty": "Pediatrician", "price": "Rs. 3,500"},
        "dr. chen": {"specialty": "Neurologist", "price": "Rs. 6,000"},
    }

    query = search_query.lower().strip()

    # Return all doctors for a general query
    if query in ["all", ""]:
        result = "Available Doctors:\n"
        for name, info in doctors.items():
            result += f"- {name.title()}: {info['specialty']} ({info['price']})\n"
        return result

    # Search by name or specialty
    for key, info in doctors.items():
        if query in key or query in info["specialty"].lower():
            return (
                f"Doctor: {key.title()}\n"
                f"Specialty: {info['specialty']}\n"
                f"Consultation Fee: {info['price']}"
            )

    return "Sorry, no doctor matches that query. Ask to see 'all' doctors."


@tool
def get_hospital_timing():
    """Get the hospital's operating hours."""
    return (
        "Hospital Timings:\n"
        "- OPD & Consultations: Monday to Saturday, 8:00 AM to 8:00 PM.\n"
        "- Emergency Services: Open 24/7."
    )


# ===============================================================
# 3. SYSTEM PROMPT
# ===============================================================
SYSTEM_PROMPT = f"""
You are a friendly and professional AI receptionist for a Healthcare Hospital.
Today's date is: {current_date}. Always use this year for new bookings.
You manage patient appointments using a SQLite database.

You can perform these tasks:
1. Book appointments.
2. Find appointments.
3. Show all appointments.
4. Update appointments.
5. Cancel appointments.
6. Give doctor details and prices.
7. Give hospital timings.

IMPORTANT RULES:
- ALWAYS use database tools when modifying appointment information.
- NEVER invent patient information, doctor names, or prices.
- Patient names must be unique.
- If the user asks to book multiple appointments, call book_appointment for each patient.
- Never book two patients with the same doctor at the same date and time. If the slot is taken, say it is already booked and ask for another suitable time.
- If the user asks to cancel multiple appointments, call cancel_appointment for each patient.
- Do NOT just tell the user that you performed the changes. Actually call the tools.
- NEVER give medical advice, diagnoses or medicine suggestions. Say that a doctor must be consulted.
- If the patient describes an emergency, tell them to call emergency services or come to the Emergency department immediately.
- After completing operations, give a short, friendly confirmation.
- Be concise.
"""

# ===============================================================
# 4. LLM (GROQ)
# ===============================================================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ===============================================================
# 5. MEMORY
# ===============================================================
memory = MemorySaver()

# ===============================================================
# 6. CREATE AGENT
# ===============================================================
agent = create_react_agent(
    model=llm,
    tools=[
        book_appointment,
        find_appointment,
        show_appointments,
        update_appointment,
        cancel_appointment,
        get_doctor_info,
        get_hospital_timing,
    ],
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)

# ===============================================================
# 7. CHAT CONFIG
# ===============================================================
config = {"configurable": {"thread_id": "hospital_patient_001"}}

# ===============================================================
# 8. START CHAT
# ===============================================================
print("\n======================================")
print("     AI HOSPITAL RECEPTIONIST")
print("======================================")
print("Type 'exit' or 'quit' to stop.")

while True:
    user_input = input("\nPatient: ")

    if user_input.lower().strip() in ["exit", "quit"]:
        print("\nGoodbye! Stay healthy.")
        break

    if not user_input.strip():
        continue

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Get the final AI message safely (handles lists and strings)
        ai_message = response["messages"][-1]

        print("\nAI:")
        if isinstance(ai_message.content, list):
            print(ai_message.content[0].get("text", ""))
        else:
            print(ai_message.content)

    except Exception as e:
        print("\nError:", e)