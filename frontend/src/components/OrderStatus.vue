<template>
  <main>
    <div class="content">
      <div class="top">
        <img src="@/assets/egglogo.png" alt="Menu" class="egglogo-image" />
        <div class="time">기준 시간&emsp;&emsp;<span class = "coco">{{ currentTime }}</span></div>
      </div>
      <div class="middle1">
        <div class="tables">
          <div v-for="table in tables" :key="table.id" class="table">
            <h3>Table    <span class="coco2">{{ table.id }}  </span>  :</h3>
            <br>
            <span class="coco3">{{ table.uses }}</span>  %
          </div>
        </div>
        <img src="@/assets/arrow2.png" class="arrow2">
        <div class="middle2">
          <div class = "info">
            <h1>좌석 선호도</h1>
            <br>
            <img src="@/assets/egg2.png" class="egg2">
            최다 이용 테이블 : 
            &ensp;<span class="coco">{{ besttable }}</span>&ensp;번 테이블
            <br><br>
            <img src="@/assets/egg2.png" class="egg2">
            최소 이용 테이블 : 
            &ensp;<span class="coco">{{ leasttable }}</span>&ensp;번 테이블
          </div>
          <!-- <div class="info">
            <h1>체류시간</h1>
            <br>
            <img src="@/assets/egg2.png" class="egg2">
            평균 고객 체류시간 : 
            &ensp;<span class="coco">{{ meantime }}</span>&ensp;분
          </div> -->
        </div>
      </div>
      <div class="middle3">
        <img src="@/assets/arrow.png" class="arrow">
        <h4>일별 기준 시간까지의 각 테이블 누적 이용률</h4>
      </div>
      <div class = "bottom">
      <h2>시간대별 이용률</h2>
      <br>
      <img src="@/assets/egg3.png" class="egg2">
      오픈 (10:30~12:00) :&ensp;<span class = "coco">{{ open }}</span> %
      <span class="spacer"></span>
      <img src="@/assets/egg3.png" class="egg2">
      미들 (12:00~16:00) :&ensp;<span class = "coco">{{ middle }}</span> %
      <span class="spacer"></span>
      <img src="@/assets/egg3.png" class="egg2">
      마감 (16:00~20:30) :&ensp;<span class = "coco">{{ last }}</span> %
      </div>
    </div>
  </main>
</template>

<script>
export default {
  data() {
    return {
      currentTime: 'hh:mm:ss',
      tables: [
        { id: 1, uses: 20 },
        { id: 2, uses: 30 },
        { id: 3, uses: 15 },
        { id: 4, uses: 15 },
        { id: 5, uses: 20 },
        { id: 6, uses: 0 }
      ],
      besttable: 2,
      leasttable: 6,
      meantime : 70,
      open : 50.0,
      middle : 30.0,
      last : 20.0,
    };
  },
  mounted() {
    this.updateTime();
    setInterval(this.updateTime, 1000);
  },
  methods: {
    updateTime() {
      const now = new Date();
      this.currentTime = now.toTimeString().slice(0, 8);
    },
  },
}
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

main {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 200px;
  background-color: #f9f9f9;
}
.content {
  text-align: center;
  margin-bottom: 4pc;
}
.top {
  margin: 30px;
  display: inline-flex;
  gap: 100px;
}
.egglogo-image {
  width: 200px;
  height: auto;
  margin-bottom: 10px;
}
.time {
  margin-block: auto;
  margin-left: 70px;
  width: 250px;
  padding: 15px;
  font-family: 'Bold';
  background-color: white;
  border-radius: 10px;
  font-size: 17px;
  box-shadow: 5px 5px 5px #cacaca;
}
.coco {
  color: #ff814b;
}
.middle1 {
  display: flex;
  justify-content: center;
  gap: 60px;
}
.info {
  margin: auto;
  margin-bottom: 20px;
  padding: 20px;
  font-family: 'Light';
  font-weight: bold;
  background-color: white;
  border-radius: 10px;
  font-size: 17px;
  box-shadow: 5px 5px 5px #cacaca;
}

.tables {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-gap: 10px;
  justify-content: center;
  margin-bottom: 10px;
}

.table {
  border: 4px solid #ff814b;
  padding: 12px;
  border-radius: 10px;
}
.coco2 {
  color: #ff814b;
  font-size: 13px;
  font-family: "Light";
}
.coco3 {
  color: #ff814b;
  font-size: 17px;
  font-family: "Bold";
}
.middle3 {
  display: inline-flex;
}
.arrow {
  width: 11px;
	height: 11px;
  margin-left:-21pc;
  margin-top: -15px;
  margin-right: 5px;
}
h4 {
  font-size: 11px;
  font-family: "Light";
  margin-top: -15px;
  margin-bottom: 20px;
}
.arrow2 {
  width: 35px;
	height: 35px;
  margin-top: 50px;
}
.bottom{
  margin-top: 30px;
  padding: 12px;
  font-family: 'Light';
  background-color: white;
  border-radius: 10px;
  font-size: 17px;
  box-shadow: 5px 5px 5px #cacaca;
  font-weight: bold;
}
.spacer {
  display: inline-block;
  width: 55px;
}
h1 {
  font-size: 20px;
  font-family: "Bold";
  margin-top: -5px;
  margin-bottom: -5px;
  text-align: left;
}
h2 {
  font-size: 20px;
  font-family: "Bold";
  margin-top: 5px;
  margin-left: 15px;
  margin-bottom: -8px;
  text-align: left;
}
h3 {
  font-size: 10px;
  font-family: "Light";
  margin-top: 0px;
  margin-right: 30px;
  margin-bottom: -10px;
  text-align: left;
}
.egg2 {
  width: 20px;
	height: auto;
  margin-top: -3px;
  margin-bottom: -4px;
  margin-right: 3px;
}
</style>
