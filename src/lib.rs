use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

// this has become a representation both for players and tiles
// possibly rename to `Party` in the future
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum Tile {
    X = 0,
    O = 1,
    Empty = 2,
}

impl Tile {
    pub fn opponent(&self) -> Tile {
        match self {
            Self::X => Self::O,
            Self::O => Self::X,
            Self::Empty => panic!("Called opponent on empty tile"),
        }
    }

    pub fn relation(&self, rhs: Tile) -> Relation {
        if *self == Tile::Empty || rhs == Tile::Empty {
            Relation::Neutral
        } else if *self == rhs {
            Relation::Same
        } else {
            Relation::Opponent
        }
    }

    pub fn opponent_iter() -> TileOpponentIter {
        TileOpponentIter::new()
    }

    pub fn from_char(char: char) -> Option<Self> {
        match char {
            'X' => Some(Self::X),
            'O' => Some(Self::O),
            '.' => Some(Self::Empty),
            _ => None,
        }
    }
}

impl Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Tile::X => 'X',
                Tile::O => 'O',
                Tile::Empty => '.',
            }
        )
    }
}

pub struct TileOpponentIter {
    cur: Tile,
}

impl TileOpponentIter {
    fn new() -> Self {
        Self { cur: Tile::Empty }
    }
}

impl Iterator for TileOpponentIter {
    type Item = Tile;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = match self.cur {
            Tile::Empty => Some(Tile::X),
            Tile::X => Some(Tile::O),
            Tile::O => None,
        };

        self.cur = match self.cur {
            Tile::Empty => Tile::X,
            Tile::X | Tile::O => Tile::O,
        };

        ret
    }
}

impl From<u8> for Tile {
    fn from(state: u8) -> Self {
        FromPrimitive::from_u8(state).expect("Invalid state")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    Same,
    Opponent,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vec2 {
    pub x: isize,
    pub y: isize,
}

impl Vec2 {
    pub fn new(x: isize, y: isize) -> Vec2 {
        Vec2 { x, y }
    }

    pub fn is_in_board(&self) -> bool {
        (0..8).contains(&self.x) && (0..8).contains(&self.y)
    }

    pub fn board_iter() -> Vec2BoardIter {
        Vec2BoardIter {
            cur: Vec2::new(0, -1),
        }
    }

    pub fn move_string(&self) -> String {
        String::from_utf8(vec![(b'a' + self.x as u8), b'1' + self.y as u8]).expect("unreachable")
    }
}

impl Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign<Vec2> for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign<Vec2> for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<isize> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: isize) -> Self::Output {
        Vec2::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2> for isize {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<isize> for Vec2 {
    fn mul_assign(&mut self, rhs: isize) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<isize> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: isize) -> Self::Output {
        Vec2::new(self.x / rhs, self.y / rhs)
    }
}

impl DivAssign<isize> for Vec2 {
    fn div_assign(&mut self, rhs: isize) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", b'a' + self.x as u8, self.y)
    }
}

pub struct Vec2BoardIter {
    cur: Vec2,
}

impl Iterator for Vec2BoardIter {
    type Item = Vec2;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == Vec2::new(7, 7) {
            return None;
        }

        if self.cur.y == 7 {
            self.cur.y = 0;
            self.cur.x += 1;
        } else {
            self.cur.y += 1;
        }

        Some(self.cur)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    state: u128,
}

impl Board {
    pub fn empty() -> Board {
        let mut board = Board { state: 0 };

        for i in 0..64 {
            board.set_raw_place(i, Tile::Empty);
        }

        board
    }

    pub fn new() -> Board {
        let mut board = Board::empty();

        board.set(Vec2::new(3, 3), Tile::O);
        board.set(Vec2::new(3, 4), Tile::X);
        board.set(Vec2::new(4, 3), Tile::X);
        board.set(Vec2::new(4, 4), Tile::O);

        board
    }

    fn get_raw_place(&self, place: usize) -> Tile {
        Tile::from(((self.state >> (place * 2)) & 0b11) as u8)
    }

    fn set_raw_place(&mut self, place: usize, tile: Tile) {
        self.state &= !(0b11 << (place * 2));
        self.state |= (tile as u128) << (place * 2);
    }

    fn raw_place(place: Vec2) -> usize {
        (place.x * 8 + place.y) as usize
    }

    pub fn get(&self, pos: Vec2) -> Tile {
        self.get_raw_place(Self::raw_place(pos))
    }

    pub fn set(&mut self, pos: Vec2, tile: Tile) {
        self.set_raw_place(Self::raw_place(pos), tile);
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..8 {
            for x in 0..8 {
                write!(
                    f,
                    "{}",
                    match self.get(Vec2::new(x, y)) {
                        Tile::X => 'X',
                        Tile::O => 'O',
                        Tile::Empty => '.',
                    }
                )?;
            }
            writeln!(f)?;
        }

        Ok(()) // ok?
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pos {
    pub board: Board,
    pub next_player: Tile,
}

impl Pos {
    pub fn new() -> Pos {
        Pos {
            board: Board::new(),
            next_player: Tile::X,
        }
    }

    // returns whether the move flipped any pieces
    fn place(&mut self, place: Vec2) -> bool {
        self.board.set(place, self.next_player);

        let mut flipped = false;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let step = Vec2::new(dx, dy);
                flipped |= self.flip_in_direction(place, step);
            }
        }

        flipped
    }

    // retuns whether any pieces were flipped
    fn flip_in_direction(&mut self, place: Vec2, step: Vec2) -> bool {
        let mut flipped = false;

        let mut cur = place + step;

        while cur.is_in_board() && self.board.get(cur) == self.next_player.opponent() {
            cur += step;
        }

        if cur.is_in_board() && self.board.get(cur) == self.next_player {
            loop {
                cur -= step;

                if cur == place {
                    break;
                }

                self.board.set(cur, self.next_player);
                flipped = true;
            }
        }

        flipped
    }

    fn switch_player(&mut self) {
        self.next_player = self.next_player.opponent();
    }

    pub fn play(&mut self, place: Vec2) {
        self.place(place);
        self.switch_player();

        if self.valid_moves().is_empty() {
            self.switch_player();

            if self.valid_moves().is_empty() {
                self.next_player = Tile::Empty;
            }
        }
    }

    pub fn play_clone(&self, place: Vec2) -> Self {
        let mut pos = *self;
        pos.play(place);
        pos
    }

    pub fn is_game_over(&self) -> bool {
        self.next_player == Tile::Empty
    }

    pub fn tile_counts(&self) -> (usize, usize) {
        let mut x_count = 0;
        let mut o_count = 0;

        for vec2 in Vec2::board_iter() {
            match self.board.get(vec2) {
                Tile::X => x_count += 1,
                Tile::O => o_count += 1,
                Tile::Empty => {}
            }
        }

        (x_count, o_count)
    }

    pub fn winner(&self) -> Tile {
        let (x_count, o_count) = self.tile_counts();

        match x_count.cmp(&o_count) {
            Ordering::Less => Tile::O,
            Ordering::Equal => Tile::Empty,
            Ordering::Greater => Tile::X,
        }
    }

    pub fn is_valid_move(&self, place: Vec2) -> bool {
        self.board.get(place) == Tile::Empty && self.clone().place(place)
    }

    fn validate(&self, valid: &mut [[bool; 8]; 8], delta: Vec2, start: Vec2) {
        #[derive(Debug, Clone, Copy)]
        enum State {
            Reset,
            Ours,
            OursOpponent,
        }

        use Relation::*;
        use State::*;

        let mut cur = start;
        let mut state = Reset;

        while cur.is_in_board() {
            let relation = self.board.get(cur).relation(self.next_player);

            match (state, relation) {
                (_, Same) => state = Ours,
                (Ours | OursOpponent, Opponent) => state = OursOpponent,
                (OursOpponent, Neutral) => {
                    state = Reset;
                    valid[cur.x as usize][cur.y as usize] = true;
                }
                _ => state = Reset,
            }

            cur += delta;
        }
    }

    pub fn valid_moves(&self) -> Vec<Vec2> {
        let mut valid = [[false; 8]; 8];

        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let delta = Vec2::new(dx, dy);

                if dx == 1 {
                    for y in 1..7 {
                        self.validate(&mut valid, delta, Vec2::new(0, y));
                    }
                }

                if dx == -1 {
                    for y in 1..7 {
                        self.validate(&mut valid, delta, Vec2::new(7, y));
                    }
                }

                if dy == 1 {
                    for x in 1..7 {
                        self.validate(&mut valid, delta, Vec2::new(x, 0));
                    }
                }

                if dy == -1 {
                    for x in 1..7 {
                        self.validate(&mut valid, delta, Vec2::new(x, 7));
                    }
                }

                if dx == 1 || dy == 1 {
                    self.validate(&mut valid, delta, Vec2::new(0, 0));
                }

                if dx == 1 || dy == -1 {
                    self.validate(&mut valid, delta, Vec2::new(0, 7));
                }

                if dx == -1 || dy == 1 {
                    self.validate(&mut valid, delta, Vec2::new(7, 0));
                }

                if dx == -1 || dy == -1 {
                    self.validate(&mut valid, delta, Vec2::new(7, 7));
                }
            }
        }

        let mut valid_moves = Vec::new();

        for x in 0..8 {
            for y in 0..8 {
                if valid[x][y] {
                    valid_moves.push(Vec2::new(x as isize, y as isize));
                }
            }
        }

        valid_moves
    }

    pub fn score_for(&self, tile: Tile) -> f32 {
        debug_assert!(self.is_game_over());

        let relation = tile.relation(self.winner());

        match relation {
            Relation::Same => 1.0,
            Relation::Neutral => 0.5,
            Relation::Opponent => 0.0,
        }
    }

    fn tree_end_impl(&self, depth: usize, vec: &mut Vec<Pos>) {
        if depth == 0 {
            vec.push(*self);
        } else {
            for mv in self.valid_moves() {
                self.play_clone(mv).tree_end_impl(depth - 1, vec);
            }
        }
    }

    pub fn tree_end(&self, depth: usize) -> Vec<Pos> {
        let mut ret = Vec::new();
        self.tree_end_impl(depth, &mut ret);
        ret
    }
}

impl Default for Pos {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_place_simple() {
        let mut a = Pos::new();
        a.play(Vec2::new(3, 2));

        let mut b = Pos::new();
        b.board.set(Vec2::new(3, 2), Tile::X);
        b.board.set(Vec2::new(3, 3), Tile::X);
        b.next_player = Tile::O;

        assert_eq!(a, b);
    }

    fn pos_count(pos: Pos, depth: usize) -> usize {
        if pos.is_game_over() {
            return 1;
        }

        if depth == 1 {
            return pos.valid_moves().len();
        }

        pos.valid_moves()
            .iter()
            .map(|mv| pos_count(pos.play_clone(*mv), depth - 1))
            .sum()
    }

    fn read_pos(s: &str) -> Pos {
        let mut pos = Pos::new();

        let lns: Vec<_> = s.split('\n').skip(1).collect();

        for row in 0..8 {
            let ln = lns[row as usize];
            let chs: Vec<_> = ln.trim().chars().collect();
            assert_eq!(chs.len(), 8, "Line length should be 8");

            for col in 0..8 {
                pos.board.set(
                    Vec2::new(col, row),
                    Tile::from_char(chs[col as usize]).expect("Invalid board character"),
                );
            }
        }

        let ln = lns[8].trim();
        debug_assert_eq!(ln.len(), 1);
        pos.next_player = match ln.chars().next().unwrap() {
            'X' => Tile::X,
            'O' => Tile::O,
            c => panic!("Invalid next player character '{}'", c),
        };

        pos
    }

    #[test]
    fn pos_count_1() {
        let pos = Pos::new();

        assert_eq!(pos_count(pos, 7), 55092);
    }

    #[test]
    fn pos_count_2() {
        let pos = read_pos(
            r#"
            OOOOOO..
            OOOOOOXX
            OOOOOXOO
            .OXOXOOO
            .OXOOOOO
            OOXOOOOO
            OOOOOX..
            XXXXX.O.
            X
            "#,
        );

        assert_eq!(pos_count(pos, 8), 2822);
    }

    #[test]
    fn pos_count_3() {
        let pos = read_pos(
            r#"
            O.......
            .O......
            O.OXXX..
            .OOOXO..
            ..OOOO..
            ..XOXX..
            ....X...
            ....X...
            X
            "#,
        );

        assert_eq!(pos_count(pos, 5), 183805);
    }

    #[test]
    fn pos_count_4() {
        let pos = read_pos(
            r#"
            ........
            ........
            ..O.....
            ..XOX...
            ...XO...
            ........
            ........
            ........
            X
            "#,
        );

        assert_eq!(pos_count(pos, 7), 200261);
    }

    #[test]
    fn pos_count_5() {
        let pos = read_pos(
            r#"
            XXXXXXXX
            XOOOOOXO
            XXXXXXOO
            XXOOXOOO
            OOOXOO..
            OOOXOO..
            OOOOOO..
            O.O..O..
            X
            "#,
        );

        assert_eq!(pos_count(pos, 7), 30980);
    }

    #[test]
    fn pos_count_6() {
        let pos = read_pos(
            r#"
            ........
            ........
            ........
            ..XXX...
            ...XO...
            ........
            ........
            ........
            O
            "#,
        );

        assert_eq!(pos_count(pos, 7), 97554);
    }

    #[test]
    fn pos_count_7() {
        let pos = read_pos(
            r#"
            .OOOOOO.
            OXXXXO.X
            OOXXXXO.
            OXOXOX.O
            OXXOXX..
            OXXOOXX.
            .XXXXXX.
            XOOOOOOO
            X
            "#,
        );

        assert_eq!(pos_count(pos, 7), 15562);
    }

    #[test]
    fn vec_mul_div() {
        let mut v = Vec2::new(3, 4);

        assert_eq!(Vec2::new(6, 8), v * 2);
        assert_eq!(Vec2::new(6, 8), 2 * v);

        v *= 2;

        assert_eq!(Vec2::new(6, 8), v);
        assert_eq!(Vec2::new(3, 4), v / 2);

        v /= 2;

        assert_eq!(Vec2::new(3, 4), v);
    }
}
