# SmartCut AI Setup & Run Guide

Follow these steps to get the full-stack SmartCut AI platform running on your local machine.

## 🚀 Prerequisites
- **Python 3.10+**
- **Node.js (v18+) & npm**
- **PostgreSQL** (Ensure a database is created and the connection string in `backend/.env` is updated)

---

## 🛠️ Backend Setup (FastAPI)

1. **Open a terminal** and navigate to the backend directory:
   ```powershell
   cd backend
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Run the Backend**:
   ```powershell
   uvicorn app.main:app --reload
   ```
   *The backend will be available at `http://localhost:8000`*

---

## 🎨 Frontend Setup (React)

1. **Open a SECOND terminal** and stay in the root project directory (`SampleSmartCUt`):
   ```powershell
   cd .. # (If you are still in the backend folder)
   ```

2. **Run the Frontend**:
   ```powershell
   npm run dev
   ```
   *The frontend will be available at `http://localhost:5173`*

---

## 📝 Configuration Note
Ensure your `backend/.env` and `src/lib/api.ts` are correctly configured:
- `backend/.env` should contain your **DATABASE_URL**.
- `src/lib/api.ts` is set to point to `http://localhost:8000/api/v1`.
