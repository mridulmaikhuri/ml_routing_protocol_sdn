# ML-Based Routing Protocol for SDN using Mininet and Python

## 📌 Project Overview
This project aims to develop a **machine learning-based routing protocol** for **Software-Defined Networking (SDN)** using **Mininet** and **Python**. The goal is to optimize network routing dynamically based on real-time traffic conditions, improving efficiency and reducing congestion.

## 🚀 Features
- **Software-Defined Networking (SDN):** Uses OpenFlow-enabled switches in Mininet.
- **Machine Learning Integration:** Predicts optimal paths for network traffic.
- **Real-Time Traffic Monitoring:** Uses packet flow statistics to train and test the model.
- **Custom SDN Controller:** Developed in Python to interact with Mininet.

## 🛠️ Technologies Used
- **Mininet** – Network emulation
- **Python** – Core programming language
- **Ryu/POX** – SDN Controller framework
- **Scikit-learn/TensorFlow** – Machine Learning algorithms
- **Wireshark** – Packet analysis

## 📂 Project Structure
```
ml-routing-sdn/
│── controller/           # Custom SDN controller
│── dataset/              # Collected network traffic data
│── models/               # Trained ML models
│── scripts/              # Python scripts for data collection & analysis
│── README.md             # Project documentation
```

## 🔧 Installation & Setup
### 1️⃣ Install Mininet
```sh
sudo apt update && sudo apt install mininet -y
```

### 2️⃣ Clone this Repository
```sh
git clone https://github.com/your-username/ml-routing-sdn.git
cd ml-routing-sdn
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Start Mininet with a Custom Topology
```sh
sudo mn --custom topology.py --controller=remote,ip=127.0.0.1 --switch=ovsk
```

### 5️⃣ Run the SDN Controller
```sh
python controller/ml_controller.py
```

### 6️⃣ Train the Machine Learning Model
```sh
python scripts/train_model.py
```

## 📝 To-Do List
✅ Basic Mininet topology setup  
✅ Collect network traffic data  
✅ Implement SDN controller  
🔲 Train and fine-tune ML model  
🔲 Evaluate routing performance  

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---
🚀 **Let’s build intelligent SDN routing together!**