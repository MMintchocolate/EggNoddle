<template>
	<div class="seat-management">
    <h1>좌석안내시스템으로 매장을 효율적으로 관리하세요</h1>
    <p class="text">부경대학교 시스템경영공학부 시나브로팀이 고안한 좌석안내시스템은 실시간으로 빈 테이블과 좌석을
			<br>
			탐지하여 매장 정보를 업데이트 합니다. 직관적이고 편리한 우리의 시스템을 이용해보세요.</p>
    
    <div class="card-container">
      <div class="card1">
				<div class="image-container">
					<img src="@/assets/sinabro_logo_2.png" class="logo-image">
					<img src="@/assets/sinabroteam.png" class="logo-team">
				</div>
        <p>일상 속에서 흔히 겪는 이슈이지만,
					누구도 세심하게 구현하지 않은 '잔여좌석 탐지'에 주목하여,
					이를 해결하기 위한 탐구를 진행했습니다. 우리 팀의 혁신적인 결과물을 직접 체험하고 그 편리함을 느껴보세요.</p>
      </div>
      <div class="card2">
				<div class="card2_1">
					<img src="@/assets/Money_Sack.png" class="Money_Sack">
          <div>
            <h2>To. 사장님!</h2>
            <p>매장을 효율적으로 운영할 수 있도록 경영적인 인사이트를 제시합니다.</p>
          </div>
        </div>
			</div>
    </div>

    <div class="card-container2">
      <div class="card3">
        <div class="card3_1">
          <div>
            <h3>To. 고객님!</h3>
            <p>매장의 잔여좌석을 실시간으로 확인하여 동선을 미리 파악하고 시간 낭비를 줄일 수 있습니다.</p>
          </div>
          <img src="@/assets/businessman.png" class="businessman"> 
        </div>
      </div>
      <div class="card4">
        <img src="@/assets/computerman.png" alt="초기 설정" class="computerman">
        <p>초기 설정을 통해 매장별로 고유한 테이블과 좌석 정보를 적용하여, 각 매장에 최적화된 정보를 제공합니다.</p>
      </div>
    </div>
  </div>
</template>

<script setup>

</script>

<style  scoped>
@font-face {
  font-family: 'Bold';
  src: url('@/assets/GmarketSansTTFBold.ttf') format('truetype');
}
@font-face {
  font-family: 'Light';
  src: url('@/assets/GmarketSansTTFLight.ttf') format('truetype');
}


.seat-management {
  text-align: center;
  padding: 20px;
  font-family: Arial, sans-serif;
  background: linear-gradient(to bottom right, #c7d7fc 0%, #a0bfff 75%);
  min-height: 200px;
}

h1 {
  font-size: 24px;
  margin-bottom: 20px;
  color: #333;
	font-family: 'Bold';
}

p {
  font-size: 15px;
  margin-bottom: 20px;
  color: #666;
	font-family: 'Light';
  font-weight: bold;
}
.text {
  font-size: 12px;
  margin-bottom: 20px;
  color: #666;
	font-family: 'Light';
  font-weight: lighter;
}
.image-container {
	display: flex;
  align-content: space-around;
}
.logo-image {
	width: 120px;
  height: auto;
	margin-right: 15px;
}
.logo-team {
	width: 80px;
  height: auto;
  border-radius: 0 !important;
	margin-bottom: 15px;
}
.card-container {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 40px;
  margin-top: 2.5pc;
  margin-bottom: -30px;
}

.card1 {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  width: 250px;
  height: auto;
  text-align: left;
}
.card2 {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  width: 400px;
  height: auto;
  text-align: left;
}
.card2_1 {
	display: flex;
}

h2 {
  font-size: 25px;
  font-family: "Light";
  margin-right: 20px;
  color: #525ace;
  text-align: right;
}
.Money_Sack {
  size: 70px;
  height: 100px;
  margin-top: 20px;
  margin-right: 20px;
}
.card-container2 {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 40px;
  margin-bottom: 2.5pc;
}
.card3 {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  max-width: 400px;
  height: auto;
  text-align: left;
}
.card4 {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  max-width: 250px;
  height: auto;
  text-align: left;
}
.card3_1 {
  display: flex;
}
.businessman {
  size: 70px;
  height: 120px;
  margin-left: 20px;
}
.computerman {
  width: 150px;
	height: auto;
  margin-left: 50px;
}
.info {
  max-width: 300px;
  text-align: left;
}

h3 {
  font-size: 25px;
  font-family: "Light";
  color: #525ace;
  text-align: left;
}

img {
  width: 150px;
	height: auto;
}
</style>