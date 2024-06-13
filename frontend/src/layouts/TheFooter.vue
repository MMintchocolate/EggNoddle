<template>
  <footer :class="footerBackgroundClass">
    <div class="footer-content-main">시나브로팀</div>
    <div class="footer-content-second">캡스톤 디자인 : 실시간 좌석 안내 시스템</div>
    <div class="footer-content-final"><br>&copy; 2024 Sinabro. All rights reserved.</div>
  </footer>
</template>

<script>
export default {
  name: 'FooterComponent',
  props: {
    currentRouteName: {
      type: String,
      required: true
    }
  },
  computed: {
    footerBackgroundClass() {
      return this.currentRouteName === 'Home' ? 'home-footer-background' : 'userstatus-footer-background';
    }
  }
};
</script>

<style scoped>
@font-face {
  font-family: 'Bold';
  src: url('@/assets/GmarketSansTTFBold.ttf') format('truetype');
}
@font-face {
  font-family: 'Light';
  src: url('@/assets/GmarketSansTTFLight.ttf') format('truetype');
}


footer {
  color: white;
  text-align: center;
  padding: 20px 0;
}
.footer-content-main {
  font-size: 25px;
  font-family: "Light";
  margin: 5px;
}
.footer-content-second {
  font-size: 15px;
  font-family: "Light";
  margin: 5px;
}
.footer-content-final {
  font-size: 10px;
  font-family: "Light";
  margin: 5px;
}
</style>
