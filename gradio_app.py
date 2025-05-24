import gradio as gr
import httpx  # For making API calls

AGENT_API_URL = "http://0.0.0.0:5055/answer"


def call_agent_api(question: str):
    """
    Calls your agent API using httpx and expects two messages in return.
    """
    params = {"question": question}

    try:
        with httpx.Client(timeout=60.0) as client:  # 30-second timeout
            response = client.get(AGENT_API_URL, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()

        reply = data.get("response", "Error: Agent did not provide the first reply.")

        return reply

    except httpx.TimeoutException:
        return "Error: The request to the agent API timed out.", ""
    except httpx.RequestError as e:
        return f"Error: Could not connect/request agent API: {e}", ""
    except httpx.HTTPStatusError as e:
        return (
            f"Error: API returned status {e.response.status_code}. Response: {e.response.text}",
            "",
        )
    except (
        KeyError,
        IndexError,
        TypeError,
    ) as e:
        return (
            f"Error: API response format issue: {e}. Response: {data if 'data' in locals() else 'N/A'}",
            "",
        )
    except Exception as e:  # Catch-all for other unexpected errors
        return f"An unexpected error occurred: {e}", ""


# --- Gradio Interface Logic ---
def chat_func(user_message, history):
    """
    Gradio function to handle the chat interaction.
    'history' is a list of [user_msg, bot_msg] pairs.
    """
    history = history or []  # Ensure history is a list

    agent_msg1 = call_agent_api(user_message)

    full_agent_response = f"{agent_msg1}"

    history.append([user_message, full_agent_response])
    return (
        history,
        "",
    )


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Simple Chat with Agent API")
    gr.Markdown(
        f"Ensure your agent API is running and accessible at: `{AGENT_API_URL}`"
    )

    chatbot = gr.Chatbot(
        [], elem_id="chatbot", label="Chat Window", bubble_full_width=False, height=500
    )

    with gr.Row():
        message_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter...",
            lines=2,
            scale=7,
        )
        submit_button = gr.Button("Send", variant="primary", scale=1)

    def clear_chat_and_inputs():
        return [], "", ""

    clear_button = gr.Button("🗑️ Clear Chat")

    submit_action = [message_input, chatbot]
    submit_output = [chatbot, message_input]

    message_input.submit(chat_func, inputs=submit_action, outputs=submit_output)
    submit_button.click(chat_func, inputs=submit_action, outputs=submit_output)

    clear_button.click(
        clear_chat_and_inputs,
        inputs=None,
        outputs=[chatbot, message_input],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
