import { createRouter, createWebHistory } from 'vue-router';
import Homeview from '../views/Home.vue';
import UserStatusView from '../views/UserStatus.vue';
import OrderStatusView from '../views/OrderStatus.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Homeview
  },
  {
		path: '/userstatus',
		name: 'UserStatus',
		component: UserStatusView,
	},
  {
		path: '/orderstatus',
		name: 'OrderStatus',
		component: OrderStatusView,
	},
];

const router = createRouter({
  history: createWebHistory('/'),
  routes
});

export default router;
