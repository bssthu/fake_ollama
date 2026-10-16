"""Desired configuration versus the sockets established at process startup."""


def listener_signature(settings):
    values = []
    for kind in ("ollama", "api"):
        values.extend((kind, item.name, item.host, item.port) for item in getattr(settings, kind + "_interfaces"))
    for kind in ("admin", "dashboard", "playground"):
        values.append((kind, getattr(settings, kind + "_listener_enabled"), getattr(settings, kind + "_host"), getattr(settings, kind + "_port")))
    return values


def effective_settings(actual, previous, desired):
    """Retired model ports fail closed; management stays on the actual socket.

    Host changes retain the old interface authentication until restart. Changes
    to credentials on unchanged listener addresses remain hot-reloadable.
    """
    updates = {}
    for kind in ("ollama", "api"):
        key = kind + "_interfaces"
        bound = {item.name: item for item in getattr(actual, key)}
        active = {item.name: item for item in getattr(previous, key)}
        interfaces = []
        for item in getattr(desired, key):
            original = bound.get(item.name)
            if original is None or original.port != item.port:
                continue
            if original.host != item.host:
                prior = active.get(item.name, original)
                item = item.model_copy(update={"host": original.host, "access_tokens": list(prior.access_tokens)})
            interfaces.append(item)
        updates[key] = interfaces
    for kind in ("admin", "dashboard", "playground"):
        updates[kind + "_host"] = getattr(actual, kind + "_host")
        updates[kind + "_port"] = getattr(actual, kind + "_port")
        enabled = getattr(actual, kind + "_listener_enabled") and getattr(desired, kind + "_listener_enabled")
        if kind == "playground" and desired.playground_port != actual.playground_port:
            enabled = False
        updates[kind + "_enabled"] = enabled
    return desired.model_copy(update=updates)
