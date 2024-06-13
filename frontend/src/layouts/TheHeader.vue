<template>
  <header>
    <router-link to="/">
      <img src="@/assets/sinabro.png" alt="Menu" class="logo-image" />
    </router-link>
    <nav>
      <router-link to="/UserStatus" active-class="active">좌석현황</router-link>
      <router-link to="/OrderStatus" active-class="active">좌석안내시스템</router-link>
    </nav>
    <button class="login-btn">로그아웃</button>
  </header>
</template>

<script>
export default {
  name: 'HeaderComponent',
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

header {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 10px 20px;
  background-color: #fff;
  border-bottom: 2px solid rgba(0, 0, 0, 0.05);
}
nav a {
  margin: 0 100px;
  text-decoration: none;
  color: #000;
  font-family: 'Light';
  font-weight: 900;
}
.login-btn {
  padding: 5px 10px;
  background-color: #eeeeee;
  border: none;
  cursor: pointer;
  font-family: 'Light';
  margin-left: 70px;
}
.logo-image {
  margin: 0 100px;
  width:150px;
  height: auto;
}
.active {
  background-color: #ff814b;
  border-radius: 10px;
  padding: 8px;
  color: white;
}
</style>