# HackTCHA 🧑‍💻

Test your CAPTCHAs against multiple industry-standard AI models to assess their resistance to automated solving.

## Setup Instructions

### Mac/Linux (Terminal)

```
python3 -m venv captcha_env
source captcha_env/bin/activate
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### Prerequisites

-   Node.js and npm
-   Git (optional)

### Installation

1. **Clone the repository** (or download and extract the ZIP):

```bash
git clone [your-repo-url]
cd [project-directory]
```

2. **Set up the React frontend**:

```bash
# Install Node dependencies
npm init -y
npm install react react-dom lucide-react
npm install --save-dev vite @vitejs/plugin-react
```

3. **Create the project structure**:

```
project_root/
├── package.json
├── vite.config.js
└── frontend/
    ├── index.html
    ├── components/
    │   └── CaptchaTester.jsx
    └── main.jsx
```

4. **Configure Vite**:
   Create `vite.config.js` in the project root:

```javascript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    root: "frontend",
    build: {
        outDir: "../dist",
    },
    resolve: {
        extensions: [".js", ".jsx"],
    },
});
```

5. **Update package.json**:
   Make sure your package.json includes these scripts:

```json
{
    "scripts": {
        "dev": "vite",
        "build": "vite build",
        "preview": "vite preview"
    }
}
```

### Running the Application

In one terminal, start server.py:

```bash
python server.py
```

Leave that running, in another terminal, run:

```bash
npm run dev
```

This will run on http://localhost:5173
