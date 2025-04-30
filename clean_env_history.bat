@echo off
echo === Backing up .env file ===
copy .env ..\env_backup.txt

echo === Removing .env from Git history ===
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all

echo === Adding .env to .gitignore ===
echo .env>>.gitignore
git add .gitignore
git commit -m "Add .env to .gitignore"

echo === Cleaning up references ===
del /s /q .git\refs\original
git reflog expire --expire=now --all
git gc --prune=now

echo === Force pushing cleaned repository to GitHub ===
git push origin --force --all

echo === Done! Check your GitHub repo to confirm ===
pause
