import streamlit as st

from src.agent import run_agent
from src.tools.appointments import list_available_slots
from src.evaluation import log_interaction, evaluate_answer
from src.memory import get_patient_context, get_patient_notes

st.set_page_config(page_title="Agentic Healthcare Assistant", layout="wide")

st.title("🩺 Agentic Healthcare Assistant")

tab1, tab2, tab3 = st.tabs(
    ["Patient / Attendant View", "Doctor / Admin View", "Agent Logs & Evaluation"]
)

# ---------------------------
# TAB 1: PATIENT / ATTENDANT
# ---------------------------
with tab1:
    st.subheader("Chat with the Assistant")

    st.markdown(
        """
        ### Examples:
        • *"My 70-year-old father has chronic kidney disease. Please book a nephrologist for him tomorrow."*  
        • *"Summarize the medical history for Anjali Mehra."*  
        • *"What are the latest treatments for Type 2 Diabetes?"*  
        • *"Update history for Anjali Mehra: she is now taking amlodipine and has hypertension."*
        """
    )

    user_input = st.text_area("Your request", height=120)

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_eval" not in st.session_state:
        st.session_state["last_eval"] = None

    if st.button("Run Assistant", type="primary"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                result = run_agent(user_input)
            answer = result["answer"]
            trace = result.get("trace", {})

            eval_result = evaluate_answer(user_input, answer)

            st.session_state["last_result"] = result
            st.session_state["last_eval"] = eval_result

            st.session_state["history"].append(("You", user_input))
            st.session_state["history"].append(("Assistant", answer))

            log_interaction(user_input, answer, trace=trace, eval_result=eval_result)

    # Latest response
    if st.session_state["last_result"] is not None:
        st.markdown("### Latest Response")
        st.markdown(st.session_state["last_result"]["answer"])

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state["history"] = []
        st.session_state["last_result"] = None
        st.session_state["last_eval"] = None
        st.rerun()

    # Conversation history
    st.markdown("### Conversation History")
    for speaker, text in st.session_state["history"]:
        if speaker == "You":
            st.markdown(f"**🧑 {speaker}:** {text}")
        else:
            st.markdown(f"**🤖 {speaker}:** {text}")

# ---------------------------
# TAB 2: DOCTOR / ADMIN VIEW
# ---------------------------
with tab2:
    st.subheader("Doctor / Admin View")

    st.markdown("#### View available slots")
    speciality_filter = st.text_input("Filter by speciality (optional)", "")
    date_filter = st.text_input("Filter by date YYYY-MM-DD (optional)", "")

    if st.button("Refresh Slots"):
        slots = list_available_slots(
            speciality_filter or None, date_filter or None
        )
        if not slots:
            st.info(
                "No available slots found with the current filter. "
                "Ensure data/records.xlsx has 'speciality', 'date', 'status' columns."
            )
        else:
            st.dataframe(slots)

    st.markdown(
        """
        > You can also directly edit **data/records.xlsx** to simulate new slots,  
        > doctors, or appointments.
        """
    )

# ---------------------------
# TAB 3: AGENT LOGS, MEMORY & PLANNING
# ---------------------------
with tab3:
    st.subheader("Agent Planning, Memory & Evaluation")

    # Show last agent trace
    st.markdown("#### Last Agent Plan & Tool Trace")
    if st.session_state.get("last_result"):
        st.json(st.session_state["last_result"].get("trace", {}))
    else:
        st.info("Run a query in the Patient tab to see the plan and tool trace here.")

    # Show automatic evaluation
    st.markdown("#### Last Answer Evaluation (LLM Judge)")
    if st.session_state.get("last_eval"):
        st.json(st.session_state["last_eval"])
    else:
        st.info("No evaluation yet. Run the assistant to generate one.")

    # Patient memory viewer
    st.markdown("#### Patient Memory & Notes")
    mem_patient = st.text_input("Enter patient name to inspect memory", "")
    if st.button("Show Patient Memory & Notes"):
        if not mem_patient.strip():
            st.warning("Please enter a patient name.")
        else:
            mem_ctx = get_patient_context(mem_patient)
            notes_ctx = get_patient_notes(mem_patient)

            if not mem_ctx and not notes_ctx:
                st.info("No stored memory or notes for this patient yet.")
            else:
                if mem_ctx:
                    st.markdown("**Stored Summaries (Memory):**")
                    st.text(mem_ctx)
                if notes_ctx:
                    st.markdown("**Manual Notes / Updates:**")
                    st.text(notes_ctx)

    # Raw log file
    st.markdown("#### Raw Agent Logs (agent_logs.txt)")
    try:
        from pathlib import Path

        log_path = Path("agent_logs.txt")
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.info("No logs yet. Interact with the assistant to generate some.")
    except Exception as e:
        st.error(f"Could not load logs: {e}")
