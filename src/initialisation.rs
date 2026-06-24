use enum_dispatch::enum_dispatch;

#[derive(Default, Debug)]
pub enum InitialisationOptions {
    Random,
    #[default]
    He,
}

#[enum_dispatch]
pub trait Initialisable {
    fn initialise(&mut self, previous_shape: (usize, usize)) -> (usize, usize);
}
