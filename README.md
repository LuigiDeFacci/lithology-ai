# How to Share the Lithology AI Tool

You have two easy ways to send this to the geologist.

## Option 1: The "Professional" Way (Web Link)
**Best for:** The geologist (no installation needed, works on phone/tablet).
**Requirement:** You need a GitHub account.

1.  **Upload files to GitHub:**
    *   Create a new repository (e.g., `lithology-ai`).
    *   Upload these files: `app.py`, `requirements.txt`, `models/` (folder), `data/` (optional, or let him upload his own).
2.  **Deploy on Streamlit Cloud (Free):**
    *   Go to [share.streamlit.io](https://share.streamlit.io/).
    *   Login with GitHub.
    *   Click "New App" -> Select your repository.
    *   Click "Deploy".
3.  **Send the Link:** You will get a URL (e.g., `lithology-ai.streamlit.app`) to send to Matheus. He just clicks and uses it.

---

## Option 2: The "Quick & Dirty" Way (Zip File)
**Best for:** Confidential data (keeps everything offline) or if you don't want to use GitHub.
**Requirement:** The geologist needs Python installed.

1.  **Zip the Folder:**
    *   Select the project folder.
    *   **IMPORTANT:** Ensure the `models` folder is inside.
2.  **Send the Zip:** Email, WeTransfer, or Pen Drive.
3.  **Instructions for Him:**
    *   "Unzip the folder."
    *   "Double click **`start_tool_for_geologist.bat`**."
    *   "Wait for the browser to open."

## What to tell him to do?
1.  Drag and drop the LAS file (`7-CAM-354-RN (2).las` or others).
2.  Wait for the **Composite Log** to appear.
3.  Look at the **Lithology Track** (last column).
    *   **Orange:** Sandstone (Prob > 30%)
    *   **Green:** Shale
    *   **Blue:** Limestone
    *   **Black:** Coal
4.  Click **"Download CSV Result"** to get the numbers.
