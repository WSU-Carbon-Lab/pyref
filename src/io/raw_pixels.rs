//! Unified bulk reader for raw BITPIX=16 pixel buffers.
//!
//! Single entry point [`read_bitpix16_be_bytes`] that returns the raw
//! big-endian byte span starting at `offset` in `path`. Callers decode
//! the bytes into `i16`/`i32`/`i64` themselves because the downstream
//! semantics (with or without `BZERO`, target element width) differ
//! between catalog ingest and in-memory image materialization.
//!
//! # Environment overrides
//!
//! When the environment variable `PYREF_DISABLE_MMAP` is unset or empty,
//! the reader maps `nbytes` starting at `offset` with
//! `memmap2::MmapOptions`, copies into an owned `Vec<u8>`, and drops the
//! mapping before returning. When `PYREF_DISABLE_MMAP` is set to any
//! non-empty value (for example `PYREF_DISABLE_MMAP=1`), the reader
//! falls back to `File::open` + `seek` + bulk `read_exact`. Returning an
//! owned `Vec<u8>` means callers never hold the mapping, so the override
//! only affects how bytes are obtained, not how they are consumed.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use memmap2::MmapOptions;

use crate::errors::FitsError;

/// Reads `nbytes` big-endian BITPIX=16 pixel bytes starting at `offset` in `path`.
///
/// Returns the raw byte buffer; conversion to `i16`/`i32`/`i64` is the caller's job.
pub fn read_bitpix16_be_bytes(
    path: &Path,
    offset: u64,
    nbytes: usize,
) -> Result<Vec<u8>, FitsError> {
    let file = File::open(path).map_err(|e| FitsError::io("raw_pixels open", e))?;
    if mmap_enabled() {
        read_via_mmap(&file, offset, nbytes)
    } else {
        read_via_seek(file, offset, nbytes)
    }
}

fn mmap_enabled() -> bool {
    match std::env::var_os("PYREF_DISABLE_MMAP") {
        Some(v) => v.is_empty(),
        None => true,
    }
}

fn read_via_mmap(file: &File, offset: u64, nbytes: usize) -> Result<Vec<u8>, FitsError> {
    let mmap = unsafe {
        MmapOptions::new()
            .offset(offset)
            .len(nbytes)
            .map(file)
            .map_err(|e| FitsError::io("raw_pixels mmap", e))?
    };
    let buf = mmap.to_vec();
    drop(mmap);
    Ok(buf)
}

fn read_via_seek(mut file: File, offset: u64, nbytes: usize) -> Result<Vec<u8>, FitsError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| FitsError::io("raw_pixels seek", e))?;
    let mut buf = vec![0u8; nbytes];
    file.read_exact(&mut buf)
        .map_err(|e| FitsError::io("raw_pixels read", e))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn write_fixture(dir: &Path, prefix: &[u8], payload: &[u8]) -> std::path::PathBuf {
        let path = dir.join("raw_pixels_fixture.bin");
        let mut f = File::create(&path).expect("create fixture file");
        f.write_all(prefix).expect("write prefix");
        f.write_all(payload).expect("write payload");
        f.sync_all().expect("sync fixture");
        path
    }

    #[test]
    fn read_bitpix16_be_bytes_reads_from_offset_default_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = EnvGuard::unset("PYREF_DISABLE_MMAP");
        let tmp = tempfile::tempdir().expect("tempdir");
        let prefix = vec![0xAAu8; 37];
        let payload: Vec<u8> = (0u8..64).collect();
        let path = write_fixture(tmp.path(), &prefix, &payload);

        let got = read_bitpix16_be_bytes(&path, prefix.len() as u64, payload.len())
            .expect("read must succeed at nonzero offset");
        assert_eq!(got, payload, "bytes at offset must match exactly");
    }

    #[test]
    fn read_bitpix16_be_bytes_honors_disable_mmap_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = EnvGuard::set("PYREF_DISABLE_MMAP", "1");
        let tmp = tempfile::tempdir().expect("tempdir");
        let prefix = vec![0x55u8; 13];
        let payload: Vec<u8> = (0u8..48).rev().collect();
        let path = write_fixture(tmp.path(), &prefix, &payload);

        let got = read_bitpix16_be_bytes(&path, prefix.len() as u64, payload.len())
            .expect("seek fallback must succeed when mmap is disabled");
        assert_eq!(got, payload, "seek-path bytes must match payload");
    }

    #[test]
    fn read_bitpix16_be_bytes_short_file_errors_via_seek_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = EnvGuard::set("PYREF_DISABLE_MMAP", "1");
        let tmp = tempfile::tempdir().expect("tempdir");
        let payload: Vec<u8> = (0u8..4).collect();
        let path = write_fixture(tmp.path(), &[], &payload);

        let err = read_bitpix16_be_bytes(&path, 0, payload.len() + 16)
            .expect_err("request past EOF must error");
        assert!(
            matches!(err.kind, crate::errors::FitsErrorKind::Io),
            "expected FitsErrorKind::Io for short file, got {:?}",
            err.kind
        );
    }
}
