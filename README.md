# 🏭 Physics-Informed Digital Twin (PIDT) for Industrial Mixing and Separation

> A smart AI system that watches over industrial machines in real time, predicts problems before they happen, and helps factories run safely and efficiently — without needing a full team of experts watching the equipment 24/7.

---

## 🌐 Live App

👉 [Click here to open the app](https://yashjamine1207-physics-informed-digital-twin-pidt-fo-app-r4uvr1.streamlit.app)

---

## 💡 Why Was This Project Built?

Imagine a large factory that mixes chemicals or separates liquids (like oil and water, or milk and cream). These processes run 24 hours a day, 7 days a week.

Now imagine:
- A pump starts vibrating slightly more than usual
- The temperature inside a tank slowly starts rising
- A mixing blade starts to wear out

Nobody notices — until the machine completely breaks down.

This causes:
- **Production stops** — no products being made
- **Expensive repairs** — emergency fixes cost far more than planned maintenance
- **Safety risks** — chemical processes can be dangerous if something goes wrong
- **Lost money** — every hour of downtime costs thousands of pounds/dollars

**This project solves exactly that problem.**

It builds a "digital twin" — think of it as a **virtual copy of the real machine** — that continuously monitors what's happening, learns how the machine normally behaves, and raises an alarm the moment something starts going wrong.

---

## 🤔 What Is a Digital Twin? (Simple Explanation)

- The **real machine** is in the factory — mixing chemicals, pumping fluids, separating materials
- The **digital twin** is a computer model that mirrors everything the real machine does
- The digital twin is trained on **the laws of physics** (how fluids flow, how energy behaves) AND **real data** from the machine
- When the real machine's behaviour starts to drift away from what the digital twin expects — **an alarm is raised**

It's like having a **super-smart virtual engineer** sitting next to every machine, 24/7, who never gets tired, never misses anything, and can spot problems weeks before they become serious.

---

## 🧠 How Does the AI Work? (No Jargon)

This project uses **two AI models** working together:

### 1. Physics-Informed Neural Network (PINN)
- This is the "brain that understands physics"
- It was trained not just on data, but also on the actual equations that describe how fluids, heat, and mixing work
- Because it understands the physics, it doesn't get confused by unusual readings — it knows what's physically possible and what isn't
- It acts like a physics teacher checking the machine's "homework"

### 2. LSTM (Long Short-Term Memory) Network
- This is the "brain that remembers patterns over time"
- It watches sequences of readings (temperature, pressure, flow rate, vibration) and learns what "normal" looks like over hours and days
- If the pattern starts changing in a suspicious way, it flags it
- It acts like a security guard who has memorised exactly how people normally walk through a building — and spots immediately when someone is acting differently

### Together:
- The PINN understands **what should be happening** based on physics
- The LSTM understands **what has been happening** based on history
- Together they decide: **"Is this machine healthy or not?"**

---

## 📊 What Does the App Actually Show?

When you open the app, you will see:

- **Live Dashboard** — Real-time readings of temperature, pressure, flow rate, vibration, and more
- **Health Score** — A simple score (0–100%) showing how healthy the machine is right now
- **Fault Detection** — If something is wrong, the app shows exactly what type of fault has been detected
- **Physics Residuals** — How far the machine's actual behaviour is from what the physics says it should be doing
- **Historical Trends** — Graphs showing how the machine has been behaving over time
- **Anomaly Alerts** — Clear red/amber/green alerts telling you when to act

---

## 🔬 What Data Does It Use?

The system monitors these key signals from the machine:

| Signal | What It Measures |
|--------|-----------------|
| Temperature | How hot the fluid/material is |
| Pressure | How much force the fluid is under |
| Flow Rate | How fast the fluid is moving |
| Vibration | How much the machine is shaking |
| Motor Current | How much electricity the motor is drawing |
| pH Level | How acidic or alkaline the mixture is |
| Viscosity | How thick or thin the fluid is |
| Particle Size | How big the particles in the mixture are |

---

## 🏆 Results — What Did the AI Achieve?

- ✅ **Fault detection accuracy: 96%+** — The AI correctly identifies machine faults 96 out of 100 times
- ✅ **False alarm rate: Very low** — The system rarely raises a false alarm, so workers don't start ignoring warnings
- ✅ **Early warning: Several hours to days in advance** — Problems are spotted long before they cause a breakdown
- ✅ **Works in real time** — The app updates continuously as new sensor data comes in
- ✅ **Handles 8 different types of faults** — Including bearing failure, pump cavitation, clogging, seal leaks, and more

---

## 💰 Why Is This Valuable for a Business?

### The Problem (in numbers):
- Unplanned machine downtime costs industrial companies an average of **£50,000–£500,000 per hour**
- 70% of equipment failures could have been **prevented** if spotted early
- Traditional monitoring systems only alert you **after** something has gone wrong

### What This System Does for a Business:

**Saves Money**
- Prevents expensive emergency repairs
- Replaces costly "fix it when it breaks" with affordable "fix it before it breaks" maintenance
- Reduces spare parts wastage by only replacing parts when actually needed

**Increases Production**
- Less downtime = more products made = more revenue
- Machines run at their best because problems are fixed early, not late

**Improves Safety**
- Chemical and industrial processes can be dangerous
- Early warnings mean workers can safely shut down equipment before it becomes a hazard
- Reduces risk of chemical spills, explosions, or equipment damage

**Reduces Need for Expert Staff**
- You don't need an expert engineer watching every machine 24/7
- One engineer can monitor dozens of machines through a single dashboard
- The AI handles the routine watching — humans only step in when needed

**Builds Confidence with Clients**
- Factories can prove to their customers that their processes are controlled and reliable
- Useful for regulatory compliance and quality certifications

---

## 🏭 Which Industries Can Use This?

This technology is directly useful in:

- **Chemical manufacturing** — mixing acids, solvents, compounds
- **Food and beverage** — mixing ingredients, separating cream, filtering liquids
- **Pharmaceuticals** — precise mixing of drug compounds
- **Oil and gas** — separating oil, water, and gas from wellheads
- **Water treatment** — separating clean water from waste
- **Paper and pulp** — mixing and separation in paper production
- **Mining** — separating valuable minerals from rock slurry

---

## 🗂️ How Is the Project Organised?
📁 Project Folder
├── app.py ← The main app (what you see in the browser)
├── requirements.txt ← List of software the app needs to run
│
├── 📁 streamlit_app/
│ ├── 📁 models/
│ │ ├── pinn_weights.pkl ← The trained Physics AI brain
│ │ ├── lstm_weights.pkl ← The trained Pattern-recognition AI brain
│ │ ├── scaler.pkl ← Data normalisation tool
│ │ └── metadata.pkl ← Supporting information for the models
│ │
│ └── 📁 data/
│ └── (sensor data files used for training and testing)
│
└── 📁 notebooks/
└── (training and analysis notebooks — how the AI was built)

## 🚀 How to Run This App Locally (Step by Step)

Even if you are not technical, here's how to run this on your own computer:

**Step 1 — Download the project**
```bash
git clone https://github.com/Yashjamine1207/physics-informed-digital-twin-pidt-for-industrial-mixing-and-separation.git
cd physics-informed-digital-twin-pidt-for-industrial-mixing-and-separation
```

**Step 2 — Install the required software**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
streamlit run app.py
```

**Step 4 — Open your browser**
- Go to: `http://localhost:8501`
- The app will open automatically

---

## 🛠️ Technologies Used

| Tool | What It Does |
|------|-------------|
| Python | The programming language everything is written in |
| TensorFlow / Keras | Used to build and train the AI models |
| Streamlit | Turns the AI into a web app anyone can use |
| Plotly | Creates the interactive charts and graphs |
| Scikit-learn | Handles data preparation and evaluation |
| NumPy / Pandas | Handles numbers and data tables |

---

## 👨‍💻 Built By

**Yash Jamine**
MSc Mechanical Engineering Design — University of Manchester
📧 [Connect on LinkedIn](https://linkedin.com/in/yashjamine)

*This project was built as part of an MSc research portfolio, combining mechanical engineering knowledge with modern AI techniques to solve a real industrial problem.*
---
## 📜 Licence

This project is for academic and portfolio purposes.
Feel free to explore, learn from it, and reach out if you'd like to collaborate.
---
The best time to fix a machine is before it breaks. This project makes that possible."
