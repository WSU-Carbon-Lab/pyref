ALTER TABLE frames ADD COLUMN zarr_shape_bucket TEXT;

CREATE INDEX idx_frames_zarr_shape_bucket_frame_index
ON frames(zarr_shape_bucket, zarr_bucket_frame_index);
