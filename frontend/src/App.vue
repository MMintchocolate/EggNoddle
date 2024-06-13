<template>
  <div :class="backgroundClass">
    <TheHeader></TheHeader>

    <TheView></TheView>

    <TheFooter :currentRouteName="$route.name"></TheFooter>
  </div>
</template>

<script>

import TheHeader from '@/layouts/TheHeader.vue';
import TheView from '@/layouts/TheView.vue';
import TheFooter from '@/layouts/TheFooter.vue';

export default {
  name: 'App',
  components: {
    TheHeader,
    TheView,
    TheFooter
  },

  computed: {
    backgroundClass() {
      return this.$route.name === 'Home' ? 'home-background' : 'userstatus-background';
    }
  }
};
</script>

<style>
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}
#app {
  text-align: center;
}

.home-background {
  background-image: url('@/assets/foot_blue.png');
  background-size: cover;
}

.userstatus-background {
  background-image: url('@/assets/foot_orange.png');
  background-size: cover;
}
</style>
