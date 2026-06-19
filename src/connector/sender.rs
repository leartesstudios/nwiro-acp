//! Outbound notification + request abstraction (Wave 1 W1-06).
//!
//! Held as `Arc<dyn ClientSender>` (NOT `Box`) because Wave-3 permission tasks
//! share it. Production wraps the v0.1.18 bounded mpsc + drainer; tests capture.
#![allow(dead_code)]

use super::error::ConnectorError;
use super::runtime::BoxFuture;
use crate::acp::messages::SessionUpdateNotification;

pub trait ClientSender: Send + Sync {
    /// Fire-and-forget; drop-on-full per the Wave 1 policy.
    fn send_update(&self, notification: SessionUpdateNotification);

    /// Request/response; await-blocks for the reply. Wave 3 body; signature now.
    fn send_request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> BoxFuture<'_, Result<serde_json::Value, ConnectorError>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct StubClient {
        updates: Arc<Mutex<Vec<SessionUpdateNotification>>>,
    }

    impl ClientSender for StubClient {
        fn send_update(&self, notification: SessionUpdateNotification) {
            self.updates.lock().unwrap().push(notification);
        }
        fn send_request(
            &self,
            _method: &str,
            _params: serde_json::Value,
        ) -> BoxFuture<'_, Result<serde_json::Value, ConnectorError>> {
            Box::pin(async { Ok(serde_json::json!({})) })
        }
    }

    #[test]
    fn arc_dyn_client_sender_captures_updates() {
        let stub = StubClient::default();
        let sender: Arc<dyn ClientSender> = Arc::new(stub.clone());
        sender.send_update(SessionUpdateNotification::content_delta(
            "s".into(),
            "hi".into(),
        ));
        assert_eq!(stub.updates.lock().unwrap().len(), 1);
    }
}
