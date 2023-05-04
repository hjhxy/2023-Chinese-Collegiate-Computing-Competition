<template>
  <div class="dashboard-container">
    <div class="header">
      <div class="choose_button" @click="screenFull"><i class="el-icon-full-screen" /></div>
    </div>
    <el-divider>系统使用日志</el-divider>
    <div class="cards">
      <el-card shadow="hover">
        <div slot="header" class="clearfix" style="display: flex;justify-content: space-between;align-items: center">
          <span style="flex: 1;font-weight: bold;font-size: 20px">检测次数</span>
          <span style="flex: 1">
            <span>今日检测：</span>
            <i class="el-icon-s-flag" style="color: red" />
            <animate-number class="numbers" from="0" :to="143" duration="2000" />
          </span>
        </div>
        <div id="todaycheck" style="width: 100%; height: 200px"/>
        <el-divider>日平均检测次数：72</el-divider>
      </el-card>
      <el-card shadow="hover">
        <div slot="header" class="clearfix" style="display: flex;justify-content: space-between;align-items: center">
          <span style="flex: 1;font-weight: bold;font-size: 20px">裂缝修复报告次数</span>
          <span style="flex: 1">
            <span>今日修复：</span>
            <i class="el-icon-s-flag" style="color: red" />
            <animate-number class="numbers" from="0" :to="93" duration="2000" />
          </span>
        </div>
        <div id="todayask" style="width: 100%; height: 200px"/>
        <el-divider>日平均修复次数：58</el-divider>
      </el-card>
      <el-card shadow="hover" @click.native="goCheckHistory">
        <div slot="header" class="clearfix" style="display: flex;justify-content: space-between;align-items: center">
          <span style="flex: 1;font-weight: bold;font-size: 20px">检测历史</span>
        </div>
        <el-table
          :data="checkList"
          style="width: 100%">
          <el-table-column
            prop="user_id"
            label="用户">
          </el-table-column>
          <el-table-column
            prop="check_type"
            label="检测类型">
          </el-table-column>
          <el-table-column
            prop="check_time"
            label="检测时间">
          </el-table-column>
        </el-table>
      </el-card>
    </div>
    <el-divider>模型检测性能</el-divider>
    <div class="echarts">
      <div class="echart">
        <h3>平均检测精度</h3>
        <el-card>
          <div
            id="base-echart"
            class="base-echart"
            style="width: 100%; height: 500px"
          />
        </el-card>
      </div>
      <div class="echart">
        <h3>平均检测速度(FPS)</h3>
        <el-card>
          <div
            id="base-echart2"
            class="base-echart"
            style="width: 100%; height: 500px"
          />
        </el-card>
      </div>
    </div>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import screenfull from 'screenfull'
import { getList } from '@/api/checklist'
export default {
  name: 'Index',
  data() {
    return {
      chart: null,
      chart2: null,
      todayask: null,
      todaycheck: null,
      checkList: []
    }
  },
  mounted() {
    this.initChart()
    this.fetchData()
  },
  methods: {
    fetchData() {
      this.listLoading = true
      getList().then(response => {
        this.checkList = response.data.items.slice(0, 3)
      }).catch(() => {
        console.log('获取数据错误')
      })
    },
    initChart() {
      this.chart = echarts.init(document.getElementById('base-echart'))
      this.chart2 = echarts.init(document.getElementById('base-echart2'))
      this.todayask = echarts.init(document.getElementById('todayask'))
      this.todaycheck = echarts.init(document.getElementById('todaycheck'))
      this.chart.setOption({
        xAxis: {
          type: 'category',
          data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [90, 85, 85, 81, 89, 90, 80],
            type: 'line'
          }
        ]
      })
      this.chart2.setOption({
        xAxis: {
          type: 'category',
          data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [50, 55, 45, 60, 45, 52, 50],
            type: 'bar'
          }
        ]
      })
      this.todayask.setOption({
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [20, 32, 91, 34, 90, 133, 100],
            type: 'line',
            areaStyle: {}
          }
        ]
      })
      this.todaycheck.setOption({
        xAxis: {
          type: 'category',
          data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            data: [120, 200, 150, 80, 70, 110, 130],
            type: 'bar'
          }
        ]
      })
      window.addEventListener('resize', () => {
        this.chart.resize()
        this.chart2.resize()
        this.todayask.resize()
        this.todaycheck.resize()
      })
    },
    screenFull() {
      screenfull.toggle()
    },
    goCheckHistory() {
      this.$router.push({
        path: '/system/checkhistory'
      })
    },
    handlerTime(time) {
      time = parseInt(time)
      return String(new Date(time)).slice(10, -18)
    }
  }
}
</script>

<style lang="scss" scoped>
.dashboard {
  &-container {
    margin: 30px;

    .header {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      .choose_button{
        width: fit-content;
        margin-left: 20px;
        padding: 10px 20px;
        border-radius: 5px;
        background-color: #0173e7;
        color: #fff;
        &:hover {
          background-color: #1687ff;
          cursor: pointer;
        }
      }
    }
    .cards{
      display: flex;
      .el-card {
        flex: 1;
        margin: 10px;
        &:hover {
          cursor: pointer;
          transform: scale(1.1);
          transition: all 0.5s;
        }
        .numbers{
          font-size: 25px;
          font-weight: 700;
          color: #ee0de1;
        }
      }
    }
    .echarts{
      width: 100%;
      display: flex;
      .echart{
        flex: 1;
        margin: 10px;
      }
    }
  }
  &-text {
    font-size: 30px;
    line-height: 46px;
  }
}

::v-deep .el-dialog {
  .mode {
    margin: 5px 0px;
    padding: 15px;
    border-radius: 5px;
    text-align: center;
    background-color: #0173e7;
    color: #fff;
    &:hover{
      cursor: pointer;
      background-color: #1687ff;
    }
  }
}
</style>
