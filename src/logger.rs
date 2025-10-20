pub trait Logger {
    fn debug(&self, msg: String);
    fn info(&self, msg: String);
    fn critical(&self, msg: String);
}
impl<T> Logger for Option<T> where T : Logger{
    fn debug(&self, msg: String) {
        if let Some(logger) = self{
            logger.debug(msg);
        }
    }

    fn info(&self, msg: String) {
        if let Some(logger) = self{
            logger.info(msg)
        }
    }

    fn critical(&self, msg: String) {
        if let Some(logger) = self{
            logger.critical(msg);
        }
    }
}
pub struct BasicLogger{}
impl Logger for BasicLogger{
    fn debug(&self, msg: String) {
        println!("Debug: {msg}")
    }

    fn info(&self, msg: String) {
        println!("Info: {msg}")
    }

    fn critical(&self, msg: String) {
        println!("Critical: {msg}")
    }
}