-- Database schema for poker table analysis

-- Table states (snapshots of the table)
CREATE TABLE IF NOT EXISTS table_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    pot_size TEXT,
    community_cards TEXT,  -- JSON array of cards
    current_street TEXT,   -- preflop, flop, turn, river
    button_position INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player states (player info at each table snapshot)
CREATE TABLE IF NOT EXISTS player_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_state_id INTEGER,
    seat INTEGER,
    name TEXT,
    stack TEXT,
    cards TEXT,  -- JSON array of cards
    action TEXT, -- fold, check, call, bet, raise, all_in
    bet_amount TEXT,
    is_active BOOLEAN,
    position TEXT, -- UTG, UTG+1, MP, MP+1, CO, BTN, SB, BB
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (table_state_id) REFERENCES table_states(id)
);

-- Player profiles (aggregated player information)
CREATE TABLE IF NOT EXISTS player_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    aggression_level TEXT, -- passive, neutral, aggressive
    vpip REAL,            -- Voluntarily Put Money In Pot %
    pfr REAL,             -- Pre-Flop Raise %
    avg_stack TEXT,
    total_hands INTEGER DEFAULT 0,
    last_seen TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hand history (complete hands)
CREATE TABLE IF NOT EXISTS hand_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT UNIQUE,
    table_state_id INTEGER,
    player_actions TEXT,  -- JSON array of actions
    final_pot TEXT,
    winner TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (table_state_id) REFERENCES table_states(id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_table_states_timestamp ON table_states(timestamp);
CREATE INDEX IF NOT EXISTS idx_player_states_table_state ON player_states(table_state_id);
CREATE INDEX IF NOT EXISTS idx_player_states_name ON player_states(name);
CREATE INDEX IF NOT EXISTS idx_player_profiles_name ON player_profiles(name);
CREATE INDEX IF NOT EXISTS idx_hand_history_hand_id ON hand_history(hand_id);
