extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config: cbindgen::Config = Default::default();

    config.include_guard = Some(String::from("CHRBF_BINDINGS_H"));
    config.namespace = Some(String::from("hrbf"));
    config.line_length = 80;
    config.tab_width = 2;
    config.language = cbindgen::Language::Cxx;

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Unable to generate bindings")
        .write_to_file("hrbf.h");
}
