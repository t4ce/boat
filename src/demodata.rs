//! Source asset owned and embedded by the Boat Blueprint.

pub struct DemoAsset {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

pub static ASSETS: [DemoAsset; 1] = [DemoAsset {
    name: "Ship",
    bytes: include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/Assets/Ship/mud.glb")),
}];
