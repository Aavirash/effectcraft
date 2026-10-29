//! The `startup` key of the config file.

use effectcraft_web::startup::{CONFIG_MAX, Startup, parse};

#[test]
fn the_three_values() {
    assert_eq!(parse(br#"{"startup":"empty"}"#), Startup::Empty);
    assert_eq!(parse(br#"{"startup":"demo"}"#), Startup::Demo);
    assert_eq!(parse(br#"{"startup":"last"}"#), Startup::Last);
}

#[test]
fn absent_or_invalid_is_the_default() {
    for bytes in
        [&b"{}"[..], b"", b"not json", b"[]", b"null", br#"{"startup":1}"#, br#"{"startup":"Empty"}"#, br#"{"startup":null}"#, br#"{"startup":["empty"]}"#]
    {
        assert_eq!(parse(bytes), Startup::Last, "{:?}", String::from_utf8_lossy(bytes));
    }
}

#[test]
fn a_config_with_other_keys_still_has_its_startup() {
    assert_eq!(parse(br#"{"remote":{"mounts":[]},"startup":"empty"}"#), Startup::Empty);
    assert_eq!(parse(br#"{"startup":"demo","unknown":{"a":1}}"#), Startup::Demo);
}

#[test]
fn an_oversized_file_is_the_default() {
    let mut big = br#"{"startup":"empty","pad":""#.to_vec();
    big.extend(std::iter::repeat_n(b'x', CONFIG_MAX));
    big.extend_from_slice(br#""}"#);
    assert_eq!(parse(&big), Startup::Last);
}

#[test]
fn what_each_value_does_on_load() {
    assert!(Startup::Last.restores() && Startup::Last.demo_when_nothing_restored());
    assert!(Startup::Empty.restores() && !Startup::Empty.demo_when_nothing_restored());
    assert!(!Startup::Demo.restores() && Startup::Demo.demo_when_nothing_restored());
}
