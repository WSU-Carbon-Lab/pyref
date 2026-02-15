use std::io::{Read, Seek};

use crate::fits::error::FitsReadError;
use crate::fits::utils::FITS_BLOCK_SIZE;
const CARD_SIZE: usize = 80;

#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub keyword: String,
    pub value: CardValue,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CardValue {
    INT(i64),
    FLOAT(f64),
    STRING(String),
    LOGICAL(bool),
    EMPTY,
}

impl CardValue {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            CardValue::INT(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            CardValue::FLOAT(v) => Some(*v),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            CardValue::INT(v) => v.to_string(),
            CardValue::FLOAT(v) => v.to_string(),
            CardValue::STRING(s) => s.clone(),
            CardValue::LOGICAL(b) => b.to_string(),
            CardValue::EMPTY => String::new(),
        }
    }
}

fn value_from_str(s: &str) -> CardValue {
    let s = s.trim();
    if s.is_empty() {
        return CardValue::EMPTY;
    }
    if let Ok(n) = s.parse::<i64>() {
        return CardValue::INT(n);
    }
    if let Ok(x) = s.parse::<f64>() {
        return CardValue::FLOAT(x);
    }
    if s == "T" {
        return CardValue::LOGICAL(true);
    }
    if s == "F" {
        return CardValue::LOGICAL(false);
    }
    CardValue::STRING(s.to_string())
}

impl Card {
    pub fn parse(card_str: &str) -> Self {
        let card_str = card_str.trim();
        if card_str.len() < 10
            || card_str.starts_with("COMMENT")
            || card_str.starts_with("HISTORY")
            || !card_str.contains('=')
        {
            return Card {
                keyword: card_str.to_string(),
                value: CardValue::EMPTY,
                comment: None,
            };
        }
        let keyword = if card_str.starts_with("HIERARCH ") {
            card_str
                .splitn(2, '=')
                .next()
                .map(|s| s.replace("HIERARCH ", "").trim_end().to_string())
                .unwrap_or_default()
        } else {
            card_str
                .splitn(2, '=')
                .next()
                .map(|s| s.trim_end().to_string())
                .unwrap_or_default()
        };
        let rest = card_str.splitn(2, '=').nth(1).unwrap_or("").trim();
        let (value_str, comment) = if let Some(idx) = rest.find(" /") {
            let v = rest[..idx].trim().replace('\'', "");
            let c = rest[idx + 2..].trim().to_string();
            (v, Some(c))
        } else {
            (rest.replace('\'', ""), None)
        };
        Card {
            keyword,
            value: value_from_str(&value_str),
            comment: comment.filter(|s| !s.is_empty()),
        }
    }

    pub fn append_continue(&mut self, card_str: &str) {
        let s = card_str
            .strip_prefix("CONTINUE  ")
            .unwrap_or(card_str)
            .trim()
            .replace('\'', "");
        let s = s.strip_suffix('&').unwrap_or(&s);
        let prev = self.value.to_string();
        let prev = prev.strip_suffix('&').unwrap_or(&prev);
        self.value = CardValue::STRING(format!("{}{}", prev, s));
    }
}

impl Default for Card {
    fn default() -> Self {
        Card {
            keyword: String::new(),
            value: CardValue::EMPTY,
            comment: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Header {
    cards: Vec<Card>,
}

impl Header {
    pub fn new() -> Self {
        Header { cards: Vec::new() }
    }

    pub fn get_card(&self, name: &str) -> Option<&Card> {
        self.cards.iter().find(|c| c.keyword == name)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Card> {
        self.cards.iter()
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.cards.iter().any(|c| c.keyword == key)
    }

    pub fn read_from_file<R: Read + Seek>(reader: &mut R) -> Result<Self, FitsReadError> {
        let mut header = Header::new();
        let mut last_card = Card::default();
        loop {
            let mut block = [0u8; FITS_BLOCK_SIZE];
            reader.read_exact(&mut block)?;
            for chunk in block.chunks(CARD_SIZE) {
                let s = String::from_utf8_lossy(chunk);
                let card_str = s.trim_end();
                if card_str.trim() == "END" {
                    if !last_card.keyword.is_empty() {
                        header.cards.push(last_card);
                    }
                    return Ok(header);
                }
                if card_str.contains("CONTINUE  ") {
                    if last_card.keyword.is_empty() {
                        return Err(FitsReadError::Parse(
                            "CONTINUE without previous card".into(),
                        ));
                    }
                    last_card.append_continue(card_str);
                    continue;
                }
                if !last_card.keyword.is_empty() {
                    header.cards.push(std::mem::take(&mut last_card));
                }
                last_card = Card::parse(card_str);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_parse_int() {
        let c = Card::parse("BZERO   =                   0");
        assert_eq!(c.keyword, "BZERO");
        assert_eq!(c.value.as_int(), Some(0));
    }

    #[test]
    fn card_parse_float() {
        let c = Card::parse("EXPOSURE=                1.5 / seconds");
        assert_eq!(c.keyword, "EXPOSURE");
        assert_eq!(c.value.as_float(), Some(1.5));
        assert_eq!(c.comment.as_deref(), Some("seconds"));
    }

    #[test]
    fn card_parse_string() {
        let c = Card::parse("DATE    = '2024-01-15'");
        assert_eq!(c.keyword, "DATE");
        assert_eq!(c.value.to_string(), "2024-01-15");
    }

    #[test]
    fn card_parse_comment_empty() {
        let c = Card::parse("COMMENT something");
        assert!(c.keyword.starts_with("COMMENT"));
        assert_eq!(c.value.to_string(), "");
    }

    #[test]
    fn header_get_card() {
        let mut h = Header::new();
        h.cards.push(Card {
            keyword: "NAXIS".to_string(),
            value: CardValue::INT(2),
            comment: None,
        });
        let c = h.get_card("NAXIS").expect("card");
        assert_eq!(c.value.as_int(), Some(2));
    }
}
