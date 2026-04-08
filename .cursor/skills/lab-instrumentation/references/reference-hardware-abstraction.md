# Hardware abstraction classes

## When a HAL pays off

- **Multiple instrument models** implement the same experiment step (e.g. three power supply families under **`set_voltage(channel, volts)`**).
- **Two transports** for one logical device (**USB** vs **Ethernet** simulation) and you want one **experiment** module.
- **Unit tests** must run without drivers: you need a **`Protocol`** or **ABC** to **fake** behind stable methods.
- **Shared cross-cutting** behavior: logging, rate limits, **mutex** around non-reentrant firmware, **idempotency** after **`clear`**.

## When to skip

- **One device**, **one script**, **one resource string**: a **single module** with **functions** and a **`with`** block is simpler than a **registry** and **abstract base**.
- **SCPI one-offs** in a notebook: keep **thin** wrappers; extract a class only when the same commands appear in **three** places.

## Shape of a good HAL

- **`Instrument` protocol** (or ABC): **`connect()`**, **`disconnect()`**, **`idn()`**, **`reset()`** if applicable, and **domain methods** (`measure_voltage`, `arm_trigger`)—not raw **`write`** on the public surface unless the layer is explicitly **low-level**.
- **`Transport` vs `ProtocolHandler`**: **transport** reads bytes; **handler** turns bytes into **typed** results. Lets you test **parsing** without **hardware**.
- **Composition over deep inheritance**: **`PowerSupply`** wraps a **`VisaSession`** or **`TcpSession`**; avoid diamond hierarchies across vendors.

## Configuration

- **Model string** or **resource string** from **config file** or **env**; validate at startup. **Fail fast** if **`idn`** does not match **expected prefix** for safety interlocks.

## Discovery and plugins

- Optional **registry** mapping **`model_id` → class`** when you ship many drivers in one package; keep registration **explicit** (import side effects are hard to test).
