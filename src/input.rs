// Action enum, keybind mapping, key name lookup

#[derive(Clone, Copy, PartialEq)]
pub enum Action {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Sprint,
    Interact,
    Jump,
    Attack,
}

pub const ALL_ACTIONS: [Action; 8] = [
    Action::MoveForward,
    Action::MoveBack,
    Action::MoveLeft,
    Action::MoveRight,
    Action::Sprint,
    Action::Interact,
    Action::Jump,
    Action::Attack,
];

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::MoveForward => "Move Forward",
            Action::MoveBack => "Move Back",
            Action::MoveLeft => "Move Left",
            Action::MoveRight => "Move Right",
            Action::Sprint => "Sprint",
            Action::Interact => "Interact",
            Action::Jump => "Jump",
            Action::Attack => "Attack",
        }
    }
}

#[derive(Clone, Copy)]
pub struct KeyBinds {
    pub binds: [(Action, usize); 8],
}

impl KeyBinds {
    pub fn default_binds() -> Self {
        KeyBinds {
            binds: [
                (Action::MoveForward, 17),  // W
                (Action::MoveBack, 31),     // S
                (Action::MoveLeft, 30),     // A
                (Action::MoveRight, 32),    // D
                (Action::Sprint, 42),       // Left Shift
                (Action::Interact, 18),     // E
                (Action::Jump, 57),         // Space
                (Action::Attack, 33),       // F
            ],
        }
    }

    pub fn key_for(&self, action: Action) -> usize {
        for &(a, sc) in &self.binds {
            if a == action { return sc; }
        }
        0
    }

    pub fn is_pressed(&self, action: Action, keys: &[bool; 256]) -> bool {
        let sc = self.key_for(action);
        sc < 256 && keys[sc]
    }

    pub fn set_key(&mut self, action: Action, scancode: usize) {
        for bind in &mut self.binds {
            if bind.0 == action {
                bind.1 = scancode;
                return;
            }
        }
    }
}

pub fn key_name(scancode: usize) -> &'static str {
    match scancode {
        1 => "Esc",
        2 => "1", 3 => "2", 4 => "3", 5 => "4", 6 => "5",
        7 => "6", 8 => "7", 9 => "8", 10 => "9", 11 => "0",
        12 => "-", 13 => "=", 14 => "Backspace", 15 => "Tab",
        16 => "Q", 17 => "W", 18 => "E", 19 => "R", 20 => "T",
        21 => "Y", 22 => "U", 23 => "I", 24 => "O", 25 => "P",
        26 => "[", 27 => "]", 28 => "Enter", 29 => "LCtrl",
        30 => "A", 31 => "S", 32 => "D", 33 => "F", 34 => "G",
        35 => "H", 36 => "J", 37 => "K", 38 => "L", 39 => ";",
        40 => "'", 41 => "`", 42 => "LShift", 43 => "\\",
        44 => "Z", 45 => "X", 46 => "C", 47 => "V", 48 => "B",
        49 => "N", 50 => "M", 51 => ",", 52 => ".", 53 => "/",
        54 => "RShift", 56 => "LAlt",
        57 => "Space",
        58 => "CapsLock",
        59 => "F1", 60 => "F2", 61 => "F3", 62 => "F4", 63 => "F5",
        64 => "F6", 65 => "F7", 66 => "F8", 67 => "F9", 68 => "F10",
        87 => "F11", 88 => "F12",
        97 => "RCtrl", 100 => "RAlt",
        103 => "Up", 105 => "Left", 106 => "Right", 108 => "Down",
        _ => "???",
    }
}
