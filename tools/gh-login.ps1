# One-time GitHub CLI login for this Windows user (used by the backup trigger task).
# Piping empty input skips the "Press Enter" prompt so it starts waiting for the
# browser authorisation straight away.
'' | & 'C:\Program Files\GitHub CLI\gh.exe' auth login --hostname github.com --git-protocol ssh --skip-ssh-key --web --scopes workflow
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
