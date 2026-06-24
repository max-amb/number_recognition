#[derive(Default, Debug)]
pub enum InitialisationOptions {
    Random,
    #[default]
    He,
}

pub trait Initialisable {
    fn initialise(self, previous_shape: (usize, usize)) -> Self; 
}
