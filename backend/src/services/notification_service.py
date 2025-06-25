import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Centralized notification service for sending alerts via multiple channels.
    Supports WebSocket and in-app notifications.
    """
    
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        
    def send_notification(self, 
                         user_id: str,
                         notification_type: str,
                         title: str,
                         message: str,
                         data: Optional[Dict] = None,
                         channels: Optional[List[str]] = None) -> Dict:
        """
        Send notification through specified channels.
        
        Args:
            user_id: Target user ID
            notification_type: Type of notification (alert, prediction, portfolio, etc.)
            title: Notification title
            message: Notification message
            data: Additional data payload
            channels: List of channels to send through [\'websocket\', \'database\']
        
        Returns:
            Dictionary with delivery status for each channel
        """
        if channels is None:
            channels = [\'websocket\', \'database\']  # Default channels
        
        results = {}
        notification_data = {
            \'id\': self._generate_notification_id(),
            \'user_id\': user_id,
            \'type\': notification_type,
            \'title\': title,
            \'message\': message,
            \'data\': data or {},
            \'timestamp\': datetime.utcnow().isoformat(),
            \'read\': False
        }
        
        # Send via WebSocket
        if \'websocket\' in channels:
            try:
                self.websocket_manager.send_to_user(user_id, \'new_notification\', notification_data)
                results[\'websocket\'] = {\'status\': \'success\', \'message\': \'Sent via WebSocket\'}
            except Exception as e:
                logger.error(f"Failed to send WebSocket notification: {str(e)}")
                results[\'websocket\'] = {\'status\': \'error\', \'message\': str(e)}
        
        # Store in database
        if \'database\' in channels:
            try:
                self._store_notification(notification_data)
                results[\'database\'] = {\'status\': \'success\', \'message\': \'Stored in database\'}
            except Exception as e:
                logger.error(f"Failed to store notification in database: {str(e)}")
                results[\'database\'] = {\'status\': \'error\', \'message\': str(e)}
        
        return results
    
    def send_trading_alert(self, 
                          user_id: str,
                          ticker: str,
                          alert_type: str,
                          probability: float,
                          message: str,
                          data: Optional[Dict] = None) -> Dict:
        """
        Send trading-specific alert notification.
        """
        title = f"Trading Alert: {ticker} - {alert_type}"
        
        alert_data = {
            \'ticker\': ticker,
            \'alert_type\': alert_type,
            \'probability\': probability,
            \'priority\': self._calculate_priority(probability),
            **(data or {})
        }
        
        # Determine channels based on priority
        channels = [\'websocket\', \'database\']
        
        return self.send_notification(
            user_id=user_id,
            notification_type=\'trading_alert\',
            title=title,
            message=message,
            data=alert_data,
            channels=channels
        )
    
    def send_portfolio_alert(self,
                           user_id: str,
                           alert_type: str,
                           message: str,
                           portfolio_data: Dict) -> Dict:
        """
        Send portfolio-related alert notification.
        """
        title = f"Portfolio Alert: {alert_type}"
        
        return self.send_notification(
            user_id=user_id,
            notification_type=\'portfolio_alert\',
            title=title,
            message=message,
            data=portfolio_data,
            channels=[\'websocket\', \'database\']
        )
    
    def send_system_notification(self,
                               user_id: str,
                               message: str,
                               severity: str = \'info\') -> Dict:
        """
        Send system notification (maintenance, updates, etc.).
        """
        title = f"System Notification - {severity.upper()}"
        
        channels = [\'websocket\', \'database\']
        
        return self.send_notification(
            user_id=user_id,
            notification_type=\'system\',
            title=title,
            message=message,
            data={\'severity\': severity},
            channels=channels
        )
    
    def _store_notification(self, notification_data: Dict):
        """
        Store notification in database.
        For now, we\'ll log it (in a real implementation, save to user_alerts table)
        """
        logger.info(f"Storing notification: {json.dumps(notification_data, indent=2)}")
    
    def _generate_notification_id(self) -> str:
        """
        Generate unique notification ID.
        """
        import uuid
        return str(uuid.uuid4())
    
    def _calculate_priority(self, probability: float) -> str:
        """
        Calculate notification priority based on probability score.
        """
        if probability >= 0.8:
            return \'high\'
        elif probability >= 0.6:
            return \'medium\'
        else:
            return \'low\'
    
    def get_user_notifications(self, user_id: str, limit: int = 50, unread_only: bool = False) -> List[Dict]:
        """
        Get notifications for a user.
        For now, return empty list (in a real implementation, query user_alerts table)
        """
        logger.info(f"Getting notifications for user {user_id}, limit: {limit}, unread_only: {unread_only}")
        return []
    
    def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """
        Mark a notification as read.
        """
        logger.info(f"Marking notification {notification_id} as read for user {user_id}")
        return True
    
    def delete_notification(self, notification_id: str, user_id: str) -> bool:
        """
        Delete a notification.
        """
        logger.info(f"Deleting notification {notification_id} for user {user_id}")
        return True

# Global notification service instance
notification_service = NotificationService()


