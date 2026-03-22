---
name: cursor-sync
description: Syncs Claude Code plugins to Cursor skills and commands directories.
---

# Cursor Sync

Syncs Claude Code plugins to Cursor's skills and commands directories. Changes to installed plugins are automatically detected and synced.

## Scripts

Run these scripts from the plugin directory:

### Manual Sync
```bash
./scripts/sync.sh
```
Performs a one-time sync of all installed Claude Code plugins to `~/.cursor/skills/` and `~/.cursor/commands/`.

### Start Watcher
```bash
./scripts/watch.sh
```
Watches for plugin changes and syncs automatically. Runs in foreground.

### Install as Service
```bash
./scripts/install-service.sh install
```
Installs a launchd service that starts on login and keeps syncing in the background.

### Check Service Status
```bash
./scripts/install-service.sh status
```

### Uninstall Service
```bash
./scripts/install-service.sh uninstall
```

## How It Works

1. Reads `~/.claude/plugins/installed_plugins.json` to find installed plugins
2. For each plugin:
   - Copies skill directories from `<install_path>/skills/` to `~/.cursor/skills/`
   - Copies command files from `<install_path>/commands/` to `~/.cursor/commands/`
3. Handles naming conflicts by prefixing with plugin name
4. Cleans up orphaned files when plugins are uninstalled

## Dependencies

- `jq` for JSON parsing: `brew install jq`
- `fswatch` for file watching: `brew install fswatch`

## Logs

Service logs are written to `~/Library/Logs/cursor-sync.log`.
