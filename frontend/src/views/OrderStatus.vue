<template>
  <div>
    <OrderStatusComponent />
  </div>
</template>

<script>
import OrderStatusComponent from '../components/OrderStatus.vue';

export default {
  name: 'OrderStatusView',
  components: {
    OrderStatusComponent,
  },
};
</script>
