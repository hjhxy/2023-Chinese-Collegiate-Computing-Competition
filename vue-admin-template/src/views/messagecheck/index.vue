<template>
  <div class="app-container">
    <div class="header">
      <el-input
        v-model="username"
        placeholder="申请人账号"
        clearable
      />
      <el-button type="primary" icon="el-icon-search" @click="searchTable">搜索</el-button>
    </div>
    <el-divider />
    <el-table
      v-loading="listLoading"
      :data="list"
      element-loading-text="Loading"
      :max-height="screenHeight"
      border
      fit
      highlight-current-row
    >
      <el-table-column align="center" label="ID" width="95">
        <template slot-scope="scope">
          {{ scope.$index }}
        </template>
      </el-table-column>
      <el-table-column label="申请人">
        <template slot-scope="scope">
          {{ scope.row.nickname }}
        </template>
      </el-table-column>
      <el-table-column label="裂缝修复时间" width="110" align="center">
        <template slot-scope="scope">
          <span>{{ scope.row.msg_type }}</span>
        </template>
      </el-table-column>
      <el-table-column label="内容" width="110" align="center">
        <template slot-scope="scope">
          {{ scope.row.msg_content }}
        </template>
      </el-table-column>
      <el-table-column class-name="status-col" label="状态" width="110" align="center">
        <template slot-scope="scope">
          <el-tag :type="scope.row.msg_status | statusFilter">{{ scope.row.msg_status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column align="center" prop="created_at" label="申请时间" width="200">
        <template slot-scope="scope">
          <i class="el-icon-time" />
          <span>{{ scope.row.request_time }}</span>
        </template>
      </el-table-column>
      <el-table-column align="center" prop="created_at" label="处理时间" width="200">
        <template slot-scope="scope">
          <i class="el-icon-time" />
          <span>{{ scope.row.handle_time }}</span>
        </template>
      </el-table-column>
      <el-table-column align="center" prop="created_at" label="操作" width="200">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="danger"
            @click="handleEdit(scope.$index, scope.row)"
          >驳回</el-button>
          <el-button
            size="mini"
            type="success"
            @click="handleDelete(scope.$index, scope.row)"
          >确定</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination
      hide-on-single-page
      background
      layout="prev, pager, next"
      :total="50"
    />
  </div>
</template>

<script>
import { getList } from '@/api/msglist'
import { getScreenHeight } from '@/utils/deviceMsg'

export default {
  filters: {
    statusFilter(status) {
      const statusMap = {
        published: 'success',
        draft: 'gray',
        deleted: 'danger'
      }
      return statusMap[status]
    }
  },
  data() {
    return {
      list: null,
      listcopy: null, // 备份数据，方便数据还原
      listLoading: true,
      username: ''
    }
  },
  computed: {
    screenHeight() {
      return getScreenHeight() - 220
    }
  },
  watch: {
    username: {
      handler(newval, oldval) {
        if (newval === '') {
          this.list = this.listcopy
        }
      },
      immediate: true
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    fetchData() {
      this.listLoading = true
      getList().then(response => {
        this.list = response.data.items
        this.listcopy = this.list
        this.listLoading = false
      }).catch(() => {
        this.listLoading = false
      })
    },
    handleEdit(index, row) {
      this.$message({
        message: '已驳回',
        type: 'success'
      })
      this.list.splice(index, 1)
    },
    handleDelete(index, row) {
      this.$message({
        message: '已批准',
        type: 'success'
      })
      this.list.splice(index, 1)
    },
    searchTable() {
      if (!this.username) return
      this.list = this.list.filter(el => {
        return el.name === this.username
      })
      console.log(this.list)
    }
  }
}
</script>

<style lang="scss">
.header {
  display: flex;
  .el-input{
    margin-right: 20px;
  }
}

.el-drawer{
  padding: 0px 20px;
}

.el-pagination{
  margin: 10px auto 0;
  float: right;
}
</style>
