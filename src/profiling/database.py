"""
OpponentDatabase: Manages all opponent profiles and handles SQLite persistence.
"""
import os
import sqlite3
from typing import Any, Dict

from .opponent_profile import OpponentProfile


class OpponentDatabase:
    def __init__(self, db_path: str = "data/opponents.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        self.cache: Dict[str, OpponentProfile] = {}

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opponents (
                    name TEXT PRIMARY KEY,
                    profile_json TEXT
                )
            """
            )

    def get_profile(self, name: str) -> OpponentProfile:
        if name in self.cache:
            return self.cache[name]
        cur = self.conn.cursor()
        cur.execute("SELECT profile_json FROM opponents WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            profile = OpponentProfile.from_json(row[0])
        else:
            profile = OpponentProfile(name)
        self.cache[name] = profile
        return profile

    def save_profile(self, profile: OpponentProfile) -> None:
        profile_json = profile.to_json()
        with self.conn:
            self.conn.execute(
                "REPLACE INTO opponents (name, profile_json) VALUES (?, ?)",
                (profile.name, profile_json),
            )
        self.cache[profile.name] = profile

    def update_profile(
        self,
        name: str,
        actions: Any,
        went_to_showdown: bool = False,
        showdown_hand: Any = None,
    ) -> None:
        profile = self.get_profile(name)
        profile.record_hand(actions, went_to_showdown, showdown_hand)
        self.save_profile(profile)

    def get_all_profiles(self) -> Dict[str, OpponentProfile]:
        cur = self.conn.cursor()
        cur.execute("SELECT name, profile_json FROM opponents")
        profiles = {}
        for name, profile_json in cur.fetchall():
            profiles[name] = OpponentProfile.from_json(profile_json)
        return profiles

    def close(self) -> None:
        self.conn.close()
