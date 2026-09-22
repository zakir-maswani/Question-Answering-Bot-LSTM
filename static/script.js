const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function addMessage(text, sender, confidence = null) {
  const msg = document.createElement("div");
  msg.classList.add("msg", sender);

  if (sender === "bot" && confidence !== null && confidence < 0.5) {
    msg.classList.add("low-confidence");
  }

  const textNode = document.createElement("span");
  textNode.textContent = text;
  msg.appendChild(textNode);

  if (sender === "bot" && confidence !== null) {
    const conf = document.createElement("span");
    conf.classList.add("confidence");
    conf.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    msg.appendChild(conf);
  }

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

async function askQuestion(question) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  return response.json();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;

  try {
    const { answer, confidence } = await askQuestion(question);
    addMessage(answer, "bot", confidence);
  } catch (err) {
    addMessage("Something went wrong reaching the server.", "bot", 0);
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});
