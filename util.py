import re


def guess_value_type(value):
    if isinstance(value, dict):
        return {
            k: guess_value_type(v)
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [
            guess_value_type(item)
            for item in value
        ]
    elif isinstance(value, tuple):
        return tuple(
            guess_value_type(item)
            for item in value
        )
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        if value in ['True', 'true']:
            return True

        if value in ['False', 'false']:
            return False

        if value in ['None', 'none', 'null', 'nil']:
            return None

        return value
    else:
        return value


variable_pattern = re.compile(r'%(\S+?)%')


def render_config(file, **context):
    def replace_variable(match):
        variable_name = match.group(1)
        if variable_name not in context:
            raise KeyError('Undefined variable key \'{}\'.'
                           .format(variable_name))
        return str(context.get(variable_name, match.group(0)))

    with open(file, 'rt', encoding='utf-8') as f:
        s = f.read()

    return variable_pattern.sub(replace_variable, s)
