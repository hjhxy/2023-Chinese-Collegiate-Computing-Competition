<template>
  <div class="app-container">
    <div class="header">
      <el-input
        v-model="username"
        placeholder="账号"
        clearable
      />
      <el-button type="primary" icon="el-icon-search" @click="searchTable">搜索</el-button>
      <el-button type="danger" icon="el-icon-edit" @click="addUser">添加用户</el-button>
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
      <el-table-column label="头像">
        <template slot-scope="scope">
          <img style="width: 60px;height: auto" :src="scope.row.avatar" title="用户名">
        </template>
      </el-table-column>
      <el-table-column label="用户名">
        <template slot-scope="scope">
          {{ scope.row.nickname }}
        </template>
      </el-table-column>
      <el-table-column label="身份">
        <template slot-scope="scope">
          {{ scope.row.permission==0?"超级管理员":"普通管理员" }}
        </template>
      </el-table-column>
      <el-table-column label="账号" align="center">
        <template slot-scope="scope">
          <span>{{ scope.row.username }}</span>
        </template>
      </el-table-column>
      <el-table-column label="密码" align="center">
        <template slot-scope="scope">
          {{ scope.row.password }}
        </template>
      </el-table-column>
      <el-table-column label="手机号" align="center">
        <template slot-scope="scope">
          {{ scope.row.phone }}
        </template>
      </el-table-column>
      <el-table-column label="注册时间" align="center">
        <template slot-scope="scope">
          {{ handlerTime(scope.row.createtime) }}
        </template>
      </el-table-column>
      <el-table-column class-name="status-col" label="账号状态" width="110" align="center">
        <template slot-scope="scope">
          <el-tag :type="scope.row.isopend | statusFilter">{{ scope.row.isopend == 1? "启用中":"已禁用" }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column align="center" prop="created_at" label="操作" width="200">
        <template slot-scope="scope">
          <el-button
            size="mini"
            @click="handleEdit(scope.$index, scope.row)"
          >{{ scope.row.isopend == 1?'禁用':'启用' }}</el-button>
          <el-button
            size="mini"
            type="danger"
            @click="handleDelete(scope.$index, scope.row)"
          >删除</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination
      hide-on-single-page
      background
      layout="prev, pager, next"
      :total="list&&list.length"
    />

    <el-drawer
      title="添加用户"
      :visible.sync="drawer"
      :before-close="handleClose"
      size="40%"
    >
      <el-form label-position="left" label-width="80px" :model="formLabelAlign">
        <el-form-item label="用户名">
          <el-input v-model="formLabelAlign.nickname" />
        </el-form-item>
        <el-form-item label="账号">
          <el-input v-model="formLabelAlign.username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="formLabelAlign.password" />
        </el-form-item>
        <el-form-item label="手机号">
          <el-input v-model="formLabelAlign.phone" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="onSubmit">立即创建</el-button>
          <el-button @click="handleClose">取消</el-button>
        </el-form-item>
      </el-form>
    </el-drawer>
  </div>
</template>

<script>
import { getList, deleteUser } from '@/api/userlist'
import { register } from '@/api/user'
import { getScreenHeight } from '@/utils/deviceMsg'

export default {
  filters: {
    statusFilter(status) {
      const statusMap = {
        1: 'success',
        0: 'gray'
      }
      return statusMap[status]
    }
  },
  data() {
    return {
      username: '',
      drawer: false,
      list: null,
      listcopy: null, // 备份数据，方便数据还原
      listLoading: true,
      formLabelAlign: {
        nickname: 'user',
        username: '',
        password: '',
        phone: '18888888888'
      }
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
      })
    },
    addUser() {
      this.drawer = true
      this.formLabelAlign.username = 'user' + parseInt(Math.random() * 100000)
    },
    handleClose(done) {
      this.$confirm('确认关闭？')
        .then(_ => {
          this.drawer = false
          done()
        })
        .catch(_ => {})
    },
    // 添加用户
    onSubmit() {
      this.drawer = false
      register(this.formLabelAlign).then(() => {
        this.$message({
          message: '添加成功',
          type: 'success'
        })
      })
    },
    handleEdit(index, row) {
      row.isopend= row.isopend==0?1:0
      this.$message({
        type:"success",
        message:"操作成功"
      })
      console.log(index, row)
      // this.list[index] = this.list[index].status === '启用中' ? '禁用' : '启用中'
    },
    handleDelete(index, row) {
      this.list.splice(index,1)
      this.$message({
        type:"success",
        message:"删除成功"
      })
      console.log(index, row)
      // deleteUser(row.username)
    },
    searchTable() {
      if (!this.username) return
      this.list = this.list.filter(el => {
        return el.name === this.username
      })
      console.log(this.list)
    },
    handlerTime(time) {
      time = parseInt(time)
      return String(new Date(time)).slice(10, -18)
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
