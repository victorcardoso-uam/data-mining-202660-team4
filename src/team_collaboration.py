# File: src/team_collaboration.py
"""
Team Collaboration & Engineering Contribution Registry
Data Mining & Modern AI Systems (IIND4417) — Session 13
"""
import datetime
TEAM_REGISTRY = {
    "cohort": "Team 4",
    "repository": "data-mining-202660-team4",
    "members": [
        {
            "name": "PIEKARZ Mathieu Xavier",
            "student_id": "00647375",
            "role": "Lead Data Engineer", # e.g., ML Engineer, Data Quality Auditor
            "assigned_reviewer": "DE TALHOUET Théophile",
            "git_feature_branch": "feature/activity-10-mathieu-piekarz",
            "preferred_ai_assistant": "GitHub Copilot in VS Code",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    ]
}
def display_team_roster():
    print(f"\n{'='*20} {TEAM_REGISTRY['cohort']} ACTIVE ROSTER {'='*20}")
    for m in TEAM_REGISTRY["members"]:
        print(f"* {m['name']} ({m['student_id']}) | Role: {m['role']} | Branch: {m['git_feature_branch']}")
    print('='*60 + '\n')

if __name__ == '__main__':
    display_team_roster()