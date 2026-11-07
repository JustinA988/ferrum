//! `ferrum` command line.
//!
//! `ferrum serve --mock` binds the mock OpenAI API on `127.0.0.1:8080` unless
//! `--bind` is set. Without `--mock` the process exits, because revision 1
//! does not connect this server to a loaded model.

#![forbid(unsafe_code)]

const DEFAULT_BIND: &str = "127.0.0.1:8080";

fn main() {
    match parse_args(std::env::args().skip(1)) {
        Err(err) => {
            eprintln!("ferrum: {err}");
            std::process::exit(2);
        }
        Ok(config) if !config.mock => {
            eprintln!("ferrum: {}", engine_missing_message());
            std::process::exit(1);
        }
        Ok(config) => {
            if let Err(err) = ferrum_server::serve(&config.bind) {
                eprintln!("ferrum: {err}");
                std::process::exit(1);
            }
        }
    }
}

fn engine_missing_message() -> String {
    format!(
        "the model engine is not in revision {}",
        ferrum_core_types::INTERFACE_REVISION
    )
}

#[derive(Debug)]
struct ServeConfig {
    mock: bool,
    bind: String,
}

#[derive(Debug, PartialEq, Eq)]
enum CliError {
    Usage,
    UnknownCommand(String),
    UnknownArgument(String),
    BindRequiresAddress,
    BindRepeated,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage => f.write_str("usage: ferrum serve --mock [--bind ADDR]"),
            Self::UnknownCommand(command) => write!(f, "unknown command: {command}"),
            Self::UnknownArgument(argument) => write!(f, "unknown argument: {argument}"),
            Self::BindRequiresAddress => f.write_str("--bind requires an address"),
            Self::BindRepeated => f.write_str("--bind specified more than once"),
        }
    }
}

fn parse_args<I, S>(args: I) -> Result<ServeConfig, CliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    match args.next().as_ref().map(AsRef::as_ref) {
        None => return Err(CliError::Usage),
        Some("serve") => {}
        Some(command) => return Err(CliError::UnknownCommand(command.to_owned())),
    }

    let mut mock = false;
    let mut bind = None;
    while let Some(arg) = args.next() {
        let arg = arg.as_ref();
        if arg == "--mock" {
            mock = true;
            continue;
        }
        if arg == "--bind" {
            let Some(address) = args.next() else {
                return Err(CliError::BindRequiresAddress);
            };
            set_bind(&mut bind, address.as_ref())?;
            continue;
        }
        if let Some(address) = arg.strip_prefix("--bind=") {
            set_bind(&mut bind, address)?;
            continue;
        }
        return Err(CliError::UnknownArgument(arg.to_owned()));
    }

    Ok(ServeConfig {
        mock,
        bind: bind.unwrap_or_else(|| DEFAULT_BIND.to_owned()),
    })
}

fn set_bind(bind: &mut Option<String>, address: &str) -> Result<(), CliError> {
    if address.is_empty() || address.starts_with('-') {
        return Err(CliError::BindRequiresAddress);
    }
    if bind.is_some() {
        return Err(CliError::BindRepeated);
    }
    *bind = Some(address.to_owned());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_defaults_the_bind_address() {
        let config = parse_args(["serve", "--mock"]).unwrap();
        assert!(config.mock);
        assert_eq!(config.bind, "127.0.0.1:8080");
    }

    #[test]
    fn bind_flag_and_equals_form() {
        let spaced = parse_args(["serve", "--bind", "127.0.0.1:9", "--mock"]).unwrap();
        assert!(spaced.mock);
        assert_eq!(spaced.bind, "127.0.0.1:9");

        let equals = parse_args(["serve", "--mock", "--bind=0.0.0.0:7"]).unwrap();
        assert_eq!(equals.bind, "0.0.0.0:7");
    }

    #[test]
    fn serve_without_mock_is_the_missing_engine() {
        let config = parse_args(["serve"]).unwrap();
        assert!(!config.mock);
        assert_eq!(
            engine_missing_message(),
            "the model engine is not in revision 1"
        );
    }

    #[test]
    fn unknown_args_and_commands_fail() {
        assert_eq!(
            parse_args(["bake"]).unwrap_err(),
            CliError::UnknownCommand("bake".into())
        );
        assert_eq!(
            parse_args(["serve", "--mock", "--wat"]).unwrap_err(),
            CliError::UnknownArgument("--wat".into())
        );
        assert!(matches!(parse_args(None::<&str>), Err(CliError::Usage)));
        assert_eq!(
            parse_args(["serve", "--bind"]).unwrap_err(),
            CliError::BindRequiresAddress
        );
        assert_eq!(
            parse_args(["serve", "--bind", "--mock"]).unwrap_err(),
            CliError::BindRequiresAddress
        );
        assert_eq!(
            parse_args(["serve", "--bind", "127.0.0.1:1", "--bind=127.0.0.1:2"]).unwrap_err(),
            CliError::BindRepeated
        );
    }
}
