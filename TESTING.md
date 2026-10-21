# Testing Nullpoint with real people

A step-by-step guide for running the first real test: you today (as `p01`), then friends, then gaze tomorrow when
the camera arrives. Every step says what you should see; if you do not see it, stop and fix it before going on.

**Contents**
- [Part A. One-time setup (about 20 minutes, today)](#part-a-one-time-setup-about-20-minutes-today)
- [Part B. Each volunteer, on their own (about 25 minutes)](#part-b-each-volunteer-on-their-own-about-25-minutes)
- [Part C. Tomorrow: the camera](#part-c-tomorrow-the-camera)
- [Part D. If the electrode rig is built](#part-d-if-the-electrode-rig-is-built)
- [Part E. End of each day](#part-e-end-of-each-day)
- [Problems and fixes](#problems-and-fixes)
- [What to say to volunteers](#what-to-say-to-volunteers)

## Where everything is saved

| What | Where | Who sees it |
|---|---|---|
| Accounts (passwords stored only as salted hashes), names, consent, health group, per-task times and actions, survey answers and quotes | `data\study\` | This laptop only (git-ignored) |
| Whisper self-test clips, only if the person agrees | `data\user\<pid>\voice\` | This laptop only (git-ignored) |
| Pseudonym (`p01`) and numbers | `runs\registry.jsonl` and then the website | Public once you commit it |

Names go only into the sign-up form (or `python -m semg.study new`). Never type a name or a health detail into
other Nullpoint fields or the "Observer notes" prompt: those can end up in the public registry.

---

## Part A. One-time setup (about 20 minutes, today)

### A1. Stop using the installed app (do not uninstall it)

The installed Nullpoint (in `%LOCALAPPDATA%\Programs\Afono Nullpoint`) is from 1 October. It is missing the newest
commands ("click two five", the gaze words), its extension is older, and it saves results in its own folder instead
of this repository. For testing we run Nullpoint from the code instead.

1. If an Afono Nullpoint window is open, close it.
2. Press `Ctrl+Shift+Esc` (Task Manager), tab **Processes**. If **Afono Nullpoint** is listed, select it and click
   **End task**.
3. You do not need to uninstall it. Just do not open it from the Start menu during testing. (If you want it gone:
   Settings, Apps, Installed apps, Afono Nullpoint, Uninstall. Your old profiles in `%LOCALAPPDATA%\Afono Nullpoint`
   are kept.)

### A2. Check the code works

1. Open **PowerShell** (Start menu, type "PowerShell").
2. Type, pressing Enter after each line:
   ```powershell
   cd D:\0_Github_Repos\sEMG
   python -m pytest -q
   ```
3. **You should see**, after about 3 minutes, a line ending in `60 passed`. Warnings are fine.
4. If anything says `failed`, stop and send me the output.

### A3. Start Nullpoint from the code

1. In the same PowerShell window:
   ```powershell
   python -m app.desktop
   ```
2. **You should see** a line like `Afono Nullpoint started http://127.0.0.1:8765/ data: D:\0_Github_Repos\sEMG`,
   and a Nullpoint window opens.
3. Check the end of that line says `data: D:\0_Github_Repos\sEMG`. If PowerShell prints nothing and only a window
   appears, the old app is still running in the background: close the window, repeat A1, and try again.
4. Leave this PowerShell window open the whole time. Closing it stops Nullpoint.

### A4. Replace the browser extension

You may have an older copy of the extension loaded. Chrome must have exactly one Afono Nullpoint extension, the one
from this repository.

1. Open Chrome. In the address bar type `chrome://extensions` and press Enter.
2. Top right: turn **Developer mode** on.
3. Look for the card **Afono Nullpoint**. If there is one, click **Details** and find **Source** (it says
   "Unpacked extension" and **Loaded from** with a folder):
   - If the folder is `D:\0_Github_Repos\sEMG\extension`: go back and click the round **reload** arrow on the card.
   - If it is any other folder (for example `...\AppData\Local\Programs\Afono Nullpoint\_internal\extension`): go
     back and click **Remove**, then **Remove** again.
   - If there are two Afono cards, remove both.
4. If you removed it (or it was never there): click **Load unpacked** (top left), go to
   `D:\0_Github_Repos\sEMG\extension`, and click **Select Folder**.
5. **You should see** one **Afono Nullpoint** card, version 0.1.0, switched on, with no red **Errors** button.
6. Pin it: click the puzzle-piece icon right of the address bar, then the pin next to **Afono Nullpoint**. The φ icon
   now sits in the toolbar.
7. Check the shortcuts: go to `chrome://extensions/shortcuts`. Under Afono Nullpoint, **Show or hide target numbers**
   should be `Alt+Shift+F` and **Open the command bar** `Alt+Shift+K`. If either is empty, click the pencil and
   press the keys.
8. If you also use Microsoft Edge, open `edge://extensions` and remove any Afono extension there, so the two
   browsers do not compete for Nullpoint.

### A5. Connect the extension to Nullpoint

1. In Chrome, open https://news.ycombinator.com in a normal tab.
2. Click the φ icon in the toolbar. **You should see** a green dot, **Connected to Nullpoint**, and a count like
   "45 targets". The box **Send page targets to Nullpoint on this computer** must be ticked.
3. Press `Alt+Shift+F` on the page. **You should see** a small number next to every link. Press it again to hide them.
4. In the Nullpoint window, click the **DOM Model** tab. **You should see** "Hacker News" and the list of targets.
5. In **Send a command**, type `scroll down` and click **Send to page**. The Hacker News tab scrolls.

### A6. Earbuds as the microphone

1. Connect your earbuds to the laptop by Bluetooth.
2. Windows: Settings, System, Sound, **Input**: choose the earbuds (for AirPods, the entry with "Hands-Free" in
   its name). Speak and watch the input level bar move.
3. Expect your earbud audio to sound worse while the microphone is on. That is Bluetooth switching modes, not a bug.

### A7. Whisper mode

1. In Nullpoint, click the **Whisper** tab.
2. Click **Start microphone**. If the window asks for the microphone, click **Allow**.
3. **You should see** the status change to "Loading the speech model..." and then **Ready: whisper-base.en on cuda**
   (about 10 to 30 seconds the first time). "on cpu" also works, just slower.
4. Tick **Listen hands-free (detects when you start and stop)**. This is what lets people keep their hands in their
   lap. Without it, nothing is heard unless someone holds **Hold to talk**.
5. Keep **Send accepted phrases to the page** ticked.
6. Untick **Use gaze from the Track Model when live snap is on** (no camera today).
7. Put the windows side by side: Chrome with Hacker News on the left two thirds of the screen, the Nullpoint window
   on the right. (Windows `Win+Left` and `Win+Right` snap them.) The Hacker News tab must stay visible: Chrome pauses
   pages in windows that are fully covered, and then commands do nothing.
8. Whisper "scroll down". **You should see** the phrase under **Last phrase** in Nullpoint, and the page scroll.
9. Press `Alt+Shift+F` so the numbers show, then whisper "click" and a number you see, digit by digit (for
   badge 25, "click two five"). **You should see** that link open. Whisper "back" to return.
10. If Nullpoint hears noise when nobody speaks, move away from fans and close the window.

### A8. The website for phones

1. Open a **second** PowerShell window (leave the first one running Nullpoint):
   ```powershell
   cd D:\0_Github_Repos\sEMG
   python -m http.server 8000 -d site --bind 0.0.0.0
   ```
2. If Windows asks whether to allow Python on networks, tick **Private networks** and click **Allow**.
3. In a third PowerShell window, type `ipconfig`. Under **Wireless LAN adapter Wi-Fi**, note the **IPv4 Address**,
   for example `192.168.1.23`.
4. On your phone, on the same Wi-Fi, open `http://192.168.1.23:8000` (your address). **You should see** the black
   page with the glass φ. On the laptop, `http://127.0.0.1:8000` shows the same.
5. Leave this window running while friends test.

Setup is done. You already have the code p01 from the command line. Give it a Nullpoint login once, so your own
results stay under p01:
```powershell
python -m semg.study account --pid p01 --username yourname
```
Then in Nullpoint choose **I have an account** and sign in. Friends sign up in Nullpoint themselves (Part B).

---

## Part B. Each volunteer, on their own (about 25 minutes)

Since 21 October, Nullpoint runs the whole session itself: sign-up and consent, the voice checks, the timed task
lists (it reads each task, times it and checks where the page ended up) and the survey. Your job is only to get the
laptop ready and stay nearby for questions.

### B1. Before each person (1 minute)

1. Nullpoint is running from the code (A3) and the Chrome extension is connected (A5).
2. Chrome: one tab on https://news.ycombinator.com, scrolled to the top, zoom 100% (`Ctrl+0`).
3. Windows on screen: Chrome on the left two thirds, the Nullpoint window on the right (A7 step 7).
4. If the previous person is still signed in, click **Sign out** at the top right of Nullpoint.
5. Earbuds: theirs (connect them and pick them as the Windows input, A6) or yours, wiped.

### B2. What they see and do

They sit down, and Nullpoint shows **Welcome to the Nullpoint study**.

1. **I am new**: full name, a username, a password twice, the consent box, and whether we may keep their voice-check
   clips. The health question is optional. **Create my account and start**. Nullpoint gives them a code (`p02`) and
   shows it at the top right. If they come back tomorrow, they choose **I have an account** and sign in.
2. The **Study** tab lists their steps on the left, ticked as they finish them:
   1. **Get ready**: which earbuds, **Turn on the microphone**, and a test ("scroll down" should scroll Hacker
      News). The **Right now** box turns green when Chrome is connected and the microphone is on.
   2. **Voice check: whisper** and 3. **Voice check: quiet voice**: Nullpoint opens the voice check, starts it by
      itself and comes back to Study when the 20 phrases are done.
   4. and 5. **Task list with the mouse** and **Task list by voice**, in the order Nullpoint shows (it alternates per
      person). By voice: they say **start**, do the task, say **done** (or **skip**); hands stay in their lap.
      With the mouse they click **Start** and **Done** in Nullpoint. Nullpoint times each task and checks it from
      the page address. **Read each task aloud to me** is optional.
   6. **Your opinion**: ten 1-to-5 questions, one about using it if a mouse hurt, four open questions.
3. At the end they click **Sign out**.

The microphone indicator at the top always shows the state: grey **Microphone off**, teal **Listening**, orange
**Hearing you** (with the level bar moving), black **Working it out**.

### B3. What you still do

- **Watch, do not help.** If they ask, say "do what you would do at home". Write down what confused them.
- **Wrong actions are not counted automatically.** Nullpoint records every action and whether each task ended in
  the right place, and counts undos itself. If you saw something odd, tell me and I will check the run.
- **The website** on their phone (A8): ask them to open it and scroll once. Their answer to the last survey
  question covers it.
- Ask whether they know someone with RSI, arthritis or limited hand use who might try it. Do not take that person's
  details here; ask them to get in touch.

### B4. Fallback: you at the keyboard

If the Study tab misbehaves, the command-line tool still works and saves the same way:
```powershell
python -m semg.study new --name "Their name"
python -m semg.study run --pid p02 --mode mouse
python -m semg.study run --pid p02 --mode whisper --device "AirPods Pro" --style quiet
python -m semg.study survey --pid p02
```
You read each task aloud and press Enter when they start and finish. Set Nullpoint's **Profile** to their code
for the voice checks.

## Part C. Tomorrow: the camera

Same people where possible, same codes. Do Part A3, A5 and A7 again first (Nullpoint from the code, extension
connected, Whisper ready), then for each person:

1. Plug in the camera. Nullpoint, **Track Model** tab, **Start camera**, allow camera access.
2. **Screen width, cm**: measure the visible screen width of the laptop with a ruler (a 14-inch laptop is about
   31 cm, 15.6-inch about 34.5 cm). **Distance to screen, cm**: about 60; measure from their eyes.
3. Click **Calibrate** and have them look at each dot until it moves (9 dots, keep the head still).
4. Click **Accuracy test** (they look at dots again). **You should see** a result in degrees. Lower is better; it
   logs itself.
5. Click **Target test** (12 targets). It logs the hit rate.
6. Click **Live snap**. Back in the **Whisper** tab, tick **Use gaze from the Track Model when live snap is on**.
7. Run the task list with gaze:
   ```powershell
   python -m semg.study run --pid p02 --mode whisper-gaze --device "AirPods Pro" --style quiet
   ```
   Now they can look at a link and just say **click**.
8. If time allows, the survey again: `python -m semg.study survey --pid p02` (the newest one counts).

## Part D. If the electrode rig is built

Only with someone who agrees to electrodes on their skin. Run the laptop on battery (unplugged from the charger)
while the rig is attached.
```powershell
python -m semg.commands record --user p01 --session 1 --port COM5 --set gaze_v1
```
Find the COM port in Device Manager, Ports (COM & LPT). Take the electrodes off and put them back on between
sessions 2 and 3, then:
```powershell
python -m semg.commands train --user p01 --set gaze_v1
python -m semg.commands eval --user p01 --set gaze_v1
```
Then the task list with `--mode silent-gaze`.

## Part E. End of each day

1. Close the Nullpoint window and press `Ctrl+C` in the website PowerShell window.
2. ```powershell
   python -m semg.study summary --log
   python -m semg.site_data
   ```
   The table shows, per person and mode: tasks done, median seconds per task, errors, undos, the hands-free time
   divided by the same person's mouse time (`vs mouse`, 1.00 = as fast as a mouse) and their usability score.
3. Back up: copy the folders `data\study` and `data\user` to a USB drive or cloud folder you control.
4. Tell me the day is done. I read `data\study` and the registry myself and write up what we learned, what broke,
   and what to fix before the next round. Commit when you are happy with it (you author the commits).

If someone asks for their data to be deleted: delete `data\study\<pid>` and `data\user\<pid>`, remove their entry
from `data\study\participants.json`, and delete their lines in `runs\registry.jsonl` (search for `"participant":
"<pid>"` and `"profile": "<pid>"`).

---

## Problems and fixes

| What you see | What to do |
|---|---|
| Popup says **Nullpoint is not running** | Nullpoint's PowerShell window was closed or crashed: run `python -m app.desktop` again. |
| `Port 8765 is used by another program` | The old app is running: Task Manager, end **Afono Nullpoint**, try again. |
| Commands show in Nullpoint but the page does not move | The Hacker News tab is covered or minimised (A7 step 7), or the extension is not ticked to send targets (A5 step 2). Click once on the Hacker News tab, then try again. |
| Numbers do not appear with `Alt+Shift+F` | Reload the page (`F5`). Check `chrome://extensions/shortcuts` (A4 step 7). |
| **Speech model failed to load** | Send me the message. CPU fallback: it still works, slower. |
| Whisper hears things when nobody talks | Quieter room, earbuds closer to the mouth, or untick **Listen hands-free** and use **Hold to talk** for the self-test only. |
| Microphone level does not move | Windows sound input (A6), then click **Stop microphone** and **Start microphone** again. |
| Phone cannot open the site | Same Wi-Fi? The firewall prompt (A8 step 2)? Try the laptop's address again with `ipconfig`. |
| A story's comment link is missing (job posts have none) | Count only real stories as "third", or use the next one. Note it. |

## What to say to volunteers

> We are testing Nullpoint, software that lets you use a computer without your hands: by whispering into earbuds
> today, and with your eyes from tomorrow. It is built for people whose hands hurt or do not work, so your honest
> frustration is the most useful thing you can give us. It takes about 25 minutes. Your name stays on my laptop;
> results are published under a code like p02. You can stop at any time, and ask me to delete your data.

Whispering into earbuds is not silent speech: people nearby can hear a whisper. Fully silent input needs the skin
sensors. Say so if anyone asks.
