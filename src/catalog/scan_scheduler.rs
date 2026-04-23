use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

pub(crate) struct WaterFillQueues<T> {
    groups: Vec<(i32, VecDeque<T>)>,
    in_flight: Vec<usize>,
    alloc: Vec<usize>,
    pool_size: usize,
}

impl<T> WaterFillQueues<T> {
    pub fn new(mut pairs: Vec<(i32, Vec<T>)>) -> Self {
        pairs.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.0.cmp(&b.0)));
        let groups: Vec<_> = pairs
            .into_iter()
            .map(|(sn, v)| (sn, VecDeque::from(v)))
            .collect();
        let n = groups.len();
        Self {
            groups,
            in_flight: vec![0; n],
            alloc: vec![0; n],
            pool_size: 0,
        }
    }

    pub fn set_pool_size(&mut self, pool_size: usize) {
        self.pool_size = pool_size;
        self.rebalance();
    }

    fn rebalance(&mut self) {
        let mut pool = self.pool_size;
        for i in 0..self.groups.len() {
            let rem = self.groups[i].1.len() + self.in_flight[i];
            if rem == 0 {
                self.alloc[i] = 0;
            } else {
                let take = rem.min(pool);
                self.alloc[i] = take;
                pool = pool.saturating_sub(take);
            }
        }
    }

    pub fn dispatch_one(&mut self) -> Option<(T, usize)> {
        self.rebalance();
        for gi in 0..self.groups.len() {
            if !self.groups[gi].1.is_empty() && self.in_flight[gi] < self.alloc[gi] {
                let t = self.groups[gi].1.pop_front().unwrap();
                self.in_flight[gi] += 1;
                return Some((t, gi));
            }
        }
        None
    }

    pub fn complete_group(&mut self, group_index: usize) {
        self.in_flight[group_index] = self.in_flight[group_index].saturating_sub(1);
        self.rebalance();
    }

    pub fn total_undone(&self) -> usize {
        self.groups
            .iter()
            .enumerate()
            .map(|(i, (_, q))| q.len() + self.in_flight[i])
            .sum()
    }
}

pub(crate) struct SharedWaterFill<T> {
    inner: Mutex<WaterFillQueues<T>>,
    cv: Condvar,
}

impl<T> SharedWaterFill<T> {
    pub fn new(queues: WaterFillQueues<T>) -> Self {
        Self {
            inner: Mutex::new(queues),
            cv: Condvar::new(),
        }
    }

    pub fn set_pool_size(&self, n: usize) {
        let mut g = self.inner.lock().unwrap();
        g.set_pool_size(n);
        self.cv.notify_all();
    }

    pub fn wait_pop(&self, cancel: Option<&std::sync::atomic::AtomicBool>) -> Option<(T, usize)> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                return None;
            }
            if let Some(item) = g.dispatch_one() {
                return Some(item);
            }
            if g.total_undone() == 0 {
                return None;
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    pub fn complete_and_notify(&self, group_index: usize) {
        let mut g = self.inner.lock().unwrap();
        g.complete_group(group_index);
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_fill_drains_all_tasks() {
        let mut q = WaterFillQueues::new(vec![
            (10i32, vec![0u8, 1]),
            (20i32, vec![0u8, 1, 2, 3, 4]),
            (30i32, vec![0u8; 10]),
        ]);
        q.set_pool_size(10);
        let mut n = 0usize;
        while let Some((_t, gi)) = q.dispatch_one() {
            q.complete_group(gi);
            n += 1;
        }
        assert_eq!(n, 17);
        assert_eq!(q.total_undone(), 0);
    }
}
