DROP INDEX IF EXISTS idx_frames_zarr_shape_bucket_frame_index;
ALTER TABLE frames DROP COLUMN zarr_bucket_frame_index;
ALTER TABLE frames DROP COLUMN zarr_shape_bucket;
