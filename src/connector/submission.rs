//! Per-prompt interject handle (Wave 1 W1-05).
//!
//! Returned alongside the event stream from `AgentRuntimeConnector::prompt`.
//! Lets a caller inject ops into a running prompt without holding the stream.
//! Wave 1 ships only `Op::Cancel`; Wave 6 grows the enum (additively).
#![allow(dead_code)]

use super::error::ConnectorError;

/// Interject channel for one running prompt.
pub struct SubmissionHandle {
    op_tx: tokio::sync::mpsc::UnboundedSender<Op>,
}

/// Ops that can be injected into a running prompt.
///
/// `#[non_exhaustive]` is future-proofing for a potential crate split — within
/// this crate it does NOT enforce exhaustiveness on local `match`es. Harmless to
/// keep; Wave 6 adds `Interrupt`, `AddAttachment`, `SpawnSubAgent`, ... here
/// without a breaking change.
#[non_exhaustive]
#[derive(Debug)]
pub enum Op {
    Cancel,
}

impl SubmissionHandle {
    pub fn new(op_tx: tokio::sync::mpsc::UnboundedSender<Op>) -> Self {
        Self { op_tx }
    }

    /// Inject an op into the running prompt. Errors only if the prompt task has
    /// already finished (channel closed).
    pub fn submit(&self, op: Op) -> Result<(), ConnectorError> {
        self.op_tx
            .send(op)
            .map_err(|_| ConnectorError::Internal("submission channel closed".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn submit_delivers_then_errors_after_close() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Op>();
        let handle = SubmissionHandle::new(tx);
        handle.submit(Op::Cancel).expect("submit while open");
        assert!(matches!(rx.recv().await, Some(Op::Cancel)));
        rx.close();
        // Receiver closed -> send fails -> mapped to Internal.
        assert!(handle.submit(Op::Cancel).is_err());
    }
}
